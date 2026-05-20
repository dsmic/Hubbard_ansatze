// Attention, this library (petscmat) must be compiled with 64 bit indexing support,
// which is not standard at the moment in the conda environment for example
#include <petscmat.h>
#include <slepceps.h>


#include <vector>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <ctime>

using uint64 = uint64_t;

//
// ================= PARAMETERS =================
//
static const int Lx = 4;
static const int Ly = 4;
static const int Ns = Lx * Ly;

static const int Nup = 7;
static const int Ndown = 7;

static const double t = 1.0;
static const double U = 8.0;

static const int k_dim = 30;
static const int k_deb = 5;
static const int k_file = 30;

using idx_t = uint64;   // <- WICHTIG: PETSc-konform
//
// ================= GLOBAL BASIS =================
//
std::vector<uint64> up_basis;
std::vector<uint64> down_basis;
std::vector<idx_t> up_lookup;
std::vector<idx_t> down_lookup;

std::vector<std::pair<idx_t,idx_t>> neighbors;

int Ndown_basis;




//
// ================= BIT OPS =================
//
inline int popcount(uint64 x)
{
    return __builtin_popcountll(x);
}

inline double parity(uint64 state, int i, int j)
{
    uint64 mask;

    if (i < j)
        mask = ((1ULL << j) - 1) ^ ((1ULL << (i + 1)) - 1);
    else
        mask = ((1ULL << i) - 1) ^ ((1ULL << (j + 1)) - 1);

    return (popcount(state & mask) & 1) ? -1.0 : 1.0;
}

inline bool hop(uint64 state, int i, int j, uint64 &out, double &sign)
{
    if (!((state >> j) & 1ULL)) return false;
    if ((state >> i) & 1ULL) return false;

    out = state;
    out ^= (1ULL << i);
    out ^= (1ULL << j);

    sign = parity(state, i, j);
    return true;
}

//
// ================= BASIS GENERATION =================
//
void make_basis(int Ns, int Nf, std::vector<uint64> &basis)
{
    std::vector<int> occ(Nf);
    std::iota(occ.begin(), occ.end(), 0);

    while (true)
    {
        uint64 state = 0;
        for (int i : occ)
            state |= (1ULL << i);

        basis.push_back(state);

        int k = Nf - 1;

        while (k >= 0 && occ[k] == Ns - (Nf - k))
            k--;

        if (k < 0)
            break;

        occ[k]++;

        for (int i = k + 1; i < Nf; i++)
            occ[i] = occ[i - 1] + 1;
    }
}

//
// ================= MATMULT (ON THE FLY) =================
//
PetscErrorCode MatMult_Hubbard(Mat A, Vec x, Vec y)
{
    const PetscScalar *vx;
    PetscScalar *vy;

    VecGetArrayRead(x, &vx);
    VecGetArray(y, &vy);

    PetscInt rstart, rend;
    MatGetOwnershipRange(A, &rstart, &rend);

    #pragma omp parallel for schedule(static)
    for (PetscInt row = rstart; row < rend; row++)
    {
        idx_t iu  = row / Ndown_basis;
        idx_t idn = row % Ndown_basis;

        uint64 up   = up_basis[iu];
        uint64 down = down_basis[idn];

        double acc = U * popcount(up & down) * vx[row];

        for (size_t k = 0; k < neighbors.size(); k++)
        {
            int i = neighbors[k].first;
            int j = neighbors[k].second;

            uint64 new_up;
            double sign;

            if (hop(up, i, j, new_up, sign))
            {
                idx_t iu2 = up_lookup[new_up];
                if (iu2 >= 0)
                    acc += -t * sign * vx[iu2 * Ndown_basis + idn];
            }

            if (hop(up, j, i, new_up, sign))
            {
                idx_t iu2 = up_lookup[new_up];
                if (iu2 >= 0)
                    acc += -t * sign * vx[iu2 * Ndown_basis + idn];
            }
        }

        for (size_t k = 0; k < neighbors.size(); k++)
        {
            int i = neighbors[k].first;
            int j = neighbors[k].second;

            uint64 new_dn;
            double sign;

            if (hop(down, i, j, new_dn, sign))
            {
                idx_t id2 = down_lookup[new_dn];
                if (id2 >= 0)
                    acc += -t * sign * vx[iu * Ndown_basis + id2];
            }

            if (hop(down, j, i, new_dn, sign))
            {
                idx_t id2 = down_lookup[new_dn];
                if (id2 >= 0)
                    acc += -t * sign * vx[iu * Ndown_basis + id2];
            }
        }

        vy[row] = acc;
    }

    VecRestoreArrayRead(x, &vx);
    VecRestoreArray(y, &vy);
    return 0;
}

//
// ================= MAIN =================
//
int main(int argc, char **argv)
{
    SlepcInitialize(&argc, &argv, NULL, NULL);
    PetscPrintf(PETSC_COMM_WORLD, "sizeof(PetscInt)  = %zu bytes\n", sizeof(PetscInt));
    PetscPrintf(PETSC_COMM_WORLD, "sizeof(PetscReal) = %zu bytes\n", sizeof(PetscReal));
    PetscPrintf(PETSC_COMM_WORLD, "sizeof(void*)     = %zu bytes\n", sizeof(void*));
    if (sizeof(PetscInt) < 8)
    {
        PetscPrintf(PETSC_COMM_WORLD, "Error: PetscInt must be 64 bits for this code to work. This indicates that Petsc was not compiled with 64-bit indexing support.\n");
        SlepcFinalize();
        return 1;
    }
    MPI_Comm comm = PETSC_COMM_WORLD;

    //
    // BASIS
    //
    make_basis(Ns, Nup, up_basis);
    make_basis(Ns, Ndown, down_basis);

    Ndown_basis = down_basis.size();

    idx_t max_state = (idx_t(1) << Ns);

    up_lookup.assign(max_state, -1);
    down_lookup.assign(max_state, -1);

    for (int i = 0; i < (int)up_basis.size(); i++)
        up_lookup[up_basis[i]] = i;

    for (int i = 0; i < (int)down_basis.size(); i++)
        down_lookup[down_basis[i]] = i;

    idx_t dim = (idx_t)up_basis.size() * (idx_t)down_basis.size();

    PetscPrintf(comm, "dimension = %d\n", dim);

    //
    // NEIGHBORS (PBC 2D square)
    //
    for (int x = 0; x < Lx; x++)
    for (int y = 0; y < Ly; y++)
    {
        int i = x + Lx * y;

        int jx = ((x + 1) % Lx) + Lx * y;
        int jy = x + Lx * ((y + 1) % Ly);

        neighbors.push_back({i, jx});
        neighbors.push_back({i, jy});
    }

    //
    // SHELL MATRIX (NO CONTEXT → SAFE)
    //
    Mat A;
    MatCreateShell(comm, dim, dim, dim, dim, NULL, &A);

    MatShellSetOperation(A, MATOP_MULT, (void(*)(void))MatMult_Hubbard);

    //
    // EPS SOLVER
    //
    EPS eps;
    EPSCreate(comm, &eps);

    EPSSetOperators(eps, A, NULL);
    EPSSetProblemType(eps, EPS_HEP);
    EPSSetDimensions(eps, k_dim, PETSC_DEFAULT, PETSC_DEFAULT);
    EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);
    EPSSetFromOptions(eps);
    EPSMonitorSet(
        eps,

        [](EPS eps,
        PetscInt its,
        PetscInt nconv,
        PetscScalar* eigr,
        PetscScalar* eigi,
        PetscReal* errest,
        PetscInt nest,
        void* ctx) -> PetscErrorCode
        {
            //
            // current time
            //

            auto now =
                std::chrono::system_clock::now();

            std::time_t now_c =
                std::chrono::system_clock::to_time_t(now);

            char buf[64];

            std::strftime(
                buf,
                sizeof(buf),
                "%Y-%m-%d %H:%M:%S",
                std::localtime(&now_c)
            );

            //
            // print monitor
            //

            PetscPrintf(
                PETSC_COMM_WORLD,
                "[%s] [iter %d] nconv=%d",
                buf,
                (int)its,
                (int)nconv
            );

            int kk = PetscMin(k_deb, nest);

            if (nest > 0)
            {
                PetscPrintf(PETSC_COMM_WORLD, " ");

                for (int i = 0; i < kk; i++)
                {
                    PetscPrintf(
                        PETSC_COMM_WORLD,
                        " eig[%d]=%.12f err[%d]=%.3e",
                        i,
                        PetscRealPart(eigr[i]),
                        i,
                        (double)errest[i]
                    );
                }

                PetscPrintf(PETSC_COMM_WORLD, "\n");
            }

            PetscPrintf(
                PETSC_COMM_WORLD,
                "\n"
            );

            return 0;
        },

        NULL,
        NULL
    );

    PetscPrintf(comm, "starting solve\n");

    EPSSolve(eps);

    PetscPrintf(comm, "solve finished\n");

    PetscInt nconv;
    EPSGetConverged(eps, &nconv);
    PetscInt nprint = PetscMin((PetscInt)k_file, nconv);

    PetscPrintf(comm, "nconv = %d\n", nconv);
    PetscPrintf(comm, "nprint = %d\n", nprint);

    Vec xr;
    MatCreateVecs(A, NULL, &xr);

    for (int i = 0; i < nconv; i++)
    {
        PetscScalar eig;
        EPSGetEigenpair(eps, i, &eig, NULL, xr, NULL);

        PetscPrintf(comm, "eig[%d] = %g\n", i, (double)eig);
    }

    FILE *fvec = fopen("evecs_f32.bin", "wb");
    FILE *feig = fopen("evals_f32.bin", "wb");

    std::vector<float> buffer(dim);

    for (int i = 0; i < nprint; i++)
    {
        PetscScalar eig;
        EPSGetEigenpair(eps, i, &eig, NULL, xr, NULL);

        //
        // eigenvalue
        //
        float eig_real = (float)PetscRealPart(eig);
        fwrite(&eig_real, sizeof(float), 1, feig);

        //
        // eigenvector
        //
        PetscScalar *arr;
        VecGetArray(xr, &arr);

        #pragma omp parallel for
        for (PetscInt j = 0; j < dim; j++)
            buffer[j] = (float)PetscRealPart(arr[j]);

        fwrite(buffer.data(), sizeof(float), dim, fvec);

        VecRestoreArray(xr, &arr);

        PetscPrintf(comm, "written vec %d\n", i);
    }

    fclose(fvec);
    fclose(feig);
    EPSDestroy(&eps);
    MatDestroy(&A);
    VecDestroy(&xr);

    SlepcFinalize();
    return 0;
}