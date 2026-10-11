#include "metal_stats_path.hpp"
#include <cassert>
#include <cstdio>
#include <cstdlib>
void clear_mpi() {
 for(const char *name:{"OMPI_COMM_WORLD_RANK","OMPI_COMM_WORLD_SIZE","PMI_RANK","PMI_SIZE","PMIX_RANK","PMIX_UNIV_SIZE","SLURM_PROCID","SLURM_NTASKS"}) unsetenv(name);
}
int main() {
 using namespace metal_stats;
 clear_mpi(); assert(path(nullptr,mpi_identity())=="VASP_METAL_STATS.json");
 setenv("OMPI_COMM_WORLD_RANK","0",1); setenv("OMPI_COMM_WORLD_SIZE","2",1);
 auto mpi=mpi_identity(); assert(mpi.known && mpi.rank==0 && mpi.size==2);
 assert(path(nullptr,mpi)=="VASP_METAL_STATS.rank0.json");
 assert(path("/a.dir/stats.json",mpi)=="/a.dir/stats.rank0.json");
 assert(path("/a.dir/.stats",mpi)=="/a.dir/.stats.rank0");
 assert(path("rank-%r/stats-%r.json",mpi)=="rank-0/stats-0.json");
 setenv("OMPI_COMM_WORLD_SIZE","1",1); assert(path(nullptr,mpi_identity())=="VASP_METAL_STATS.json");
 unsetenv("OMPI_COMM_WORLD_SIZE"); assert(path("stats",mpi_identity())=="stats.rank0");
 setenv("OMPI_COMM_WORLD_RANK","not-a-rank",1); setenv("PMI_RANK","3",1); setenv("PMI_SIZE","4",1);
 mpi=mpi_identity(); assert(mpi.known && mpi.rank==3 && path(nullptr,mpi)=="VASP_METAL_STATS.rank3.json");
 clear_mpi(); setenv("SLURM_PROCID","1",1); setenv("SLURM_NTASKS","2",1);
 assert(path("stats.json",mpi_identity())=="stats.rank1.json");
 clear_mpi(); setenv("PMI_RANK","9999999999999999999999999999",1);
 assert(!mpi_identity().known); clear_mpi();
 assert(json_string("quote\"slash\\line\n\x01")=="\"quote\\\"slash\\\\line\\n\\u0001\"");
 std::puts("MPI stats paths and JSON strings passed");
}
