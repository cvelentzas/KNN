#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <time.h>
#include <cstdlib>
#include <chrono>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <sys/stat.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

// Point Structure, used in Training Dataset
typedef struct point_struct
{
	unsigned long long id;		// Point ID
	double x, y, z;				// Dimensions
} point_struct;

typedef struct partition_range //
{
	double x1, x2;			// Range
} partition_range;

// KNN Structure, used in KNN list
typedef struct knn_distance_struct
{
	unsigned long long id;		// ID of Training Point
	double distance;				// Distance between Training and Query Point
} knn_distance_struct;

typedef struct state_struct
{
	unsigned int inserted;		// Buffer Inserted Items
	double maxdist;				// Maximum Distance
} state_struct;


// Functor used for x sorting
struct x_compare {
	__host__ __device__
		bool operator()(const point_struct& o1, const point_struct& o2) {
		return o1.x < o2.x;
	}
};

// Functor used for x sorting
struct dist_compare {
	__host__ __device__
		bool operator()(const knn_distance_struct& o1, const knn_distance_struct& o2) {
		return o1.distance < o2.distance;
	}
};

//Check file existance
inline bool fileexists(const std::string& name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}

bool is_number(const std::string& s)
{
	return(strspn(s.c_str(), "-.0123456789") == s.size());
}

long GetFileSize(std::string filename)
{
	struct stat stat_buf;
	int rc = stat(filename.c_str(), &stat_buf);
	return rc == 0 ? stat_buf.st_size : -1;
}

// Template structure to pass to kernel
template <typename T>
struct KernelArray
{
	T* _array;
	int _size;
};

// Function to convert device_vector to structure
template <typename T>
KernelArray<T> convertToKernel(thrust::device_vector<T>& dVec)
{
	KernelArray<T> kArray;
	kArray._array = thrust::raw_pointer_cast(&dVec[0]);
	kArray._size = (int)dVec.size();

	return kArray;
}



__global__//k_d_training_data, k_d_query_data, k_query_distance_list KernelArray<partition_range> k_d_xpart
void runKNN(unsigned int queryPointsPerStream, unsigned int queryPointsOffest,
	KernelArray<point_struct> k_d_training_data,
	KernelArray<point_struct> k_d_query_data,
	KernelArray<knn_distance_struct> k_query_distance_list,
	KernelArray<unsigned int> k_d_inserted_list,
	KernelArray<partition_range> k_d_xpart,
	unsigned int trainingpoints, unsigned int querypoints, unsigned int distancepoints, unsigned int partitions,
	double maxx, double maxy, double maxz, unsigned int partitions_capacity_points, unsigned int k)
{
	unsigned int queryIdx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int queryIdxGlobal =queryPointsOffest + blockIdx.x * blockDim.x + threadIdx.x;
	if (queryIdxGlobal < querypoints) {
		unsigned int distance_list_offset = queryIdxGlobal * k;
		double kdsdistance;
		unsigned int inserted = 0;
		double maxdist = DBL_MAX;
		unsigned int maxid;

		inserted = k_d_inserted_list._array[queryIdxGlobal];

		if (inserted == k) {
			maxdist = 0;
			for (unsigned int j = 0; j < k; j++) {
				if (maxdist < k_query_distance_list._array[distance_list_offset + j].distance) {
					maxdist = k_query_distance_list._array[distance_list_offset + j].distance;
				}
			}
		}
		
		unsigned int cpart;

		for (cpart = 0; (cpart < k_d_xpart._size) && (k_d_xpart._array[cpart].x2 < k_d_query_data._array[queryIdx].x); cpart++) {
		}
		
		unsigned int loopnum = 0;

		double leftmost_x = 0;
		double rightmost_x = maxx;
		leftmost_x = k_d_xpart._array[cpart].x1;
		rightmost_x = k_d_xpart._array[cpart].x2;

		unsigned int leftIdx, rightIdx;
		
		
		while ((maxdist > (k_d_query_data._array[queryIdx].x - leftmost_x) * (k_d_query_data._array[queryIdx].x - leftmost_x) && (cpart >= loopnum))
			||
			((rightmost_x - k_d_query_data._array[queryIdx].x) * (rightmost_x - k_d_query_data._array[queryIdx].x) < maxdist) && (cpart + loopnum + 1 <= k_d_xpart._size))
		{
			//Find next partition to check
			if (loopnum == 0) {
				leftIdx = cpart;
				rightIdx = cpart;
			}
			else {
				//Right dist if bigger, next partition is left
				if (rightmost_x - k_d_query_data._array[queryIdx].x > k_d_query_data._array[queryIdx].x - leftmost_x) {
					if (leftIdx > 0) {
						leftIdx--;
						cpart = leftIdx;
					}
					else if (rightIdx < k_d_xpart._size - 1) {
						rightIdx++;
						cpart = rightIdx;
					}
					else break;
				}
				//Left dist if bigger, next partition is right
				else {
					if (rightIdx < k_d_xpart._size - 1) {
						rightIdx++;
						cpart = rightIdx;
					}
					else if (leftIdx > 0) {
						leftIdx--;
						cpart = leftIdx;
					}
					else break;
				}
			}
			k_d_xpart._array[cpart].x2);
			
			//Calculate Partition Points Distance and find knn points
			unsigned int id1 = cpart * partitions_capacity_points;
			unsigned int id2 = (cpart + 1) * partitions_capacity_points;
			if (id2 > trainingpoints) {
				id2 = trainingpoints;
			}
			for (unsigned int i = id1; i < id2; i++) {
				kdsdistance = (k_d_training_data._array[i].x - k_d_query_data._array[queryIdx].x) * (k_d_training_data._array[i].x - k_d_query_data._array[queryIdx].x) +
					(k_d_training_data._array[i].y - k_d_query_data._array[queryIdx].y) * (k_d_training_data._array[i].y - k_d_query_data._array[queryIdx].y) +
					(k_d_training_data._array[i].z - k_d_query_data._array[queryIdx].z) * (k_d_training_data._array[i].z - k_d_query_data._array[queryIdx].z);
				
				if (inserted < k) {
					k_query_distance_list._array[distance_list_offset + inserted].distance = kdsdistance;
					k_query_distance_list._array[distance_list_offset + inserted].id = k_d_training_data._array[i].id;
					if (maxdist < kdsdistance) {
						maxdist = kdsdistance;
						maxid = inserted;
					}
					inserted++;
				}
				else if (maxdist > kdsdistance) {
					maxdist = 0;
					for (unsigned int j = 0; j < k; j++) {
						if (maxdist < k_query_distance_list._array[distance_list_offset + j].distance) {
							maxdist = k_query_distance_list._array[distance_list_offset + j].distance;
							maxid = distance_list_offset + j;
						}
					}
					k_query_distance_list._array[maxid].distance = kdsdistance;
					k_query_distance_list._array[maxid].id = k_d_training_data._array[i].id;
					maxdist = 0;
					for (unsigned int j = 0; j < k; j++) {
						if (maxdist < k_query_distance_list._array[distance_list_offset + j].distance) {
							maxdist = k_query_distance_list._array[distance_list_offset + j].distance;
						}
					}

				}
			}
			

			leftmost_x = k_d_xpart._array[leftIdx].x1;
			rightmost_x = k_d_xpart._array[rightIdx].x2;

			loopnum++;
		}
		k_d_inserted_list._array[queryIdxGlobal] = inserted;
	}

}


// Main program
int main(int argc, char* argv[])
{
	auto timenow = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

	// Default values
	unsigned long long trainpoints = 10000;
	unsigned long long querypoints = 20;
	unsigned int k = 10;
	unsigned int testrepetitions = 100;
	unsigned int partitionsize = 1000;
	long trainingfilesize;
	long queryfilesize;
	
	bool onelineoutpout = false;
	std::string trainfilename("");
	std::string queryfilename("");
	unsigned int partitions_capacity_points = 65536;
	unsigned int NUM_STREAMS=10;

	if (argc > 1) trainfilename = argv[1];
	if (argc > 2) queryfilename = argv[2];
	if (argc > 3) k = (unsigned)atoi(argv[3]);
	if (argc > 4) partitionsize = (unsigned)atoi(argv[4]);
	if (argc > 5) testrepetitions = (unsigned)atoi(argv[5]);
	if (argc > 6) onelineoutpout = true;
	if (argc > 7) NUM_STREAMS = (unsigned)atoi(argv[7]);
	partitions_capacity_points = partitionsize / 10;

	if (!onelineoutpout)
		printf("Experiment Starting: %s", ctime(&timenow));

	const double maxx = 1;	// X axis range 0-1000
	const double maxy = 1;	// Y axis range 0-1000
	const double maxz = 1;	// Z axis range 0-1000

	// Initialize random number generator
	srand(time(NULL));
	
	trainingfilesize = GetFileSize(trainfilename);
	trainpoints = trainingfilesize / sizeof(point_struct);
	queryfilesize = GetFileSize(queryfilename);
	querypoints = queryfilesize / sizeof(point_struct);

	if (!onelineoutpout) {
		printf("Training point:%llu \tQuery points:%llu\n", trainpoints, querypoints);
	}

	unsigned int partitions = 10; 
	unsigned int backetsize;
	unsigned int partitionloops = trainpoints / partitionsize;
	unsigned int querybacketsize;
	unsigned int querypartitionsize=1024*NUM_STREAMS;
	
	// Training Host Vector
	thrust::host_vector<point_struct> training_data(partitionsize);

	// Query Vector
	thrust::host_vector<point_struct> query_data(querypartitionsize);

	thrust::device_vector<point_struct> d_query_data[NUM_STREAMS];				// Query points Device Vector, copies data from Host Vector

	thrust::device_vector<point_struct> d_training_data(partitionsize); 		// Training points Device Vector, copies data from Host Vector


	thrust::device_vector<knn_distance_struct> knn_list(k * querypoints);			// List of KNN
	
	
	KernelArray<point_struct> k_d_query_data[NUM_STREAMS];
	for(unsigned int i=0;i<NUM_STREAMS;i++) {
		d_query_data[i].resize(querypartitionsize);
		k_d_query_data[i]= convertToKernel(d_query_data[i]);
	}

	thrust::host_vector<knn_distance_struct> query_distance_list(k * querypoints,{ 0,-1 });
	
	thrust::device_vector<knn_distance_struct> d_query_distance_list = query_distance_list;
	KernelArray<knn_distance_struct> k_d_query_distance_list = convertToKernel(d_query_distance_list);


	thrust::host_vector<unsigned int> inserted_list(querypoints,0);

	thrust::device_vector<unsigned int> d_inserted_list;


	thrust::host_vector<partition_range> xpart;
	thrust::device_vector<partition_range> d_xpart;


	unsigned int pnum = 0;

	unsigned int s;

	cudaStream_t streams[NUM_STREAMS];
	for (int i = 0; i < NUM_STREAMS; i++) {
		cudaStreamCreate(&streams[i]);
	}
	unsigned int queryPointsPerStream = querypoints / NUM_STREAMS;
	if (querypoints % NUM_STREAMS > 0) queryPointsPerStream++;
	unsigned int queryPointsOffest = 0;
	// Start clock
	auto start = std::chrono::high_resolution_clock::now();
	// Loop all Training Dataset Points
	for (int r = 0; r < testrepetitions; r++) {
		d_query_distance_list = query_distance_list;
		KernelArray<knn_distance_struct> k_d_query_distance_list = convertToKernel(d_query_distance_list);
		d_inserted_list = inserted_list;
		KernelArray<unsigned int> k_d_inserted_list = convertToKernel(d_inserted_list);
		//training_data.clear();
		std::ifstream  rfile(trainfilename, std::ifstream::in | std::ifstream::binary);
		if (!rfile.is_open()) {
			printf("Error opening %s\n", trainfilename.c_str());
			return -1;
		}
		// Read all partitions
		for (unsigned int p = 0; p < partitionloops; p++) {
			if (p * partitionsize < trainpoints) {
				backetsize = partitionsize;
			}
			else {
				backetsize = p * partitionsize - trainpoints - sizeof(point_struct);
			}
			if (!rfile.read((char*)&training_data[0], backetsize * sizeof(point_struct))) {
				printf("Error reading %s, partition loop %d\n", trainfilename.c_str(), p);
				return -1;
			}


			d_training_data = training_data;
			// Sort by distance
			thrust::sort(d_training_data.begin(), d_training_data.end(), x_compare());
			training_data = d_training_data;
			

			xpart.clear();
			double prevx = 0;
			unsigned int xpart_idx;
			unsigned int i = 0;
			for (i = 0; i < partitions; i++) {
				((i + 1) * partitions_capacity_points - 1 < training_data.size()) ? xpart_idx = (i + 1) * partitions_capacity_points - 1 : xpart_idx = training_data.size() - 1;
				double x = training_data[xpart_idx].x;
				xpart.push_back({ prevx,x });
				prevx = x;
			}
			pnum++;
			d_xpart = xpart;
			KernelArray<partition_range> k_d_xpart = convertToKernel(d_xpart);

			
			KernelArray<point_struct> k_d_training_data = convertToKernel(d_training_data);

			queryPointsOffest = 0;
			unsigned int q = 0;
			std::ifstream  qfile(queryfilename, std::ifstream::in | std::ifstream::binary);
			if (!qfile.is_open()) {
				printf("Error opening %s\n", queryfilename.c_str());
				return -1;
			}
			while (queryPointsOffest<querypoints) {
				s=q % NUM_STREAMS; 

				if ((q+1) * querypartitionsize < querypoints) {
					querybacketsize = querypartitionsize;
				}
				else {
					querybacketsize = querypoints - q*querypartitionsize;
				}
				unsigned int rf=querybacketsize * sizeof(point_struct);
				if (!qfile.read((char*)&query_data[0], rf)) {
					printf("Error reading %s, partition loop %d\n", queryfilename.c_str(), q);
					return -1;
				}
				
				d_query_data[s]=query_data;

				k_d_query_data[s]= convertToKernel(d_query_data[s]);
				if (s==0 && q>0) 
					cudaDeviceSynchronize();
				runKNN <<<(querybacketsize - 1) / 64 + 1, 64, 0, streams[s] >>> (querybacketsize, queryPointsOffest,
						k_d_training_data,
						k_d_query_data[s],
						k_d_query_distance_list,
						k_d_inserted_list,
						k_d_xpart,
						trainpoints, querypoints, k * querypoints, partitions,
						maxx, maxy, maxz, partitions_capacity_points, k);
				
				queryPointsOffest += querybacketsize;
				q++;
			}

			
			cudaDeviceSynchronize();
			qfile.close();
			
		}
		rfile.close();

		
	}
	// Finish time
	auto finish = std::chrono::high_resolution_clock::now();

	if (std::string(argv[6]) == "p") {
		query_distance_list = d_query_distance_list;
		cudaDeviceSynchronize();
		for (unsigned int i = 0; i < d_query_distance_list.size(); i++) {
			printf("Query point=%u, Reference point id|\t%llu|\t%0.18f\n", i / k + 1, query_distance_list[i].id, query_distance_list[i].distance);
		}
	}

	// Print Results
	printf("Method: Disk SPP+\t ");
	printf("Training points: %llu,\t ", trainpoints);
	printf("Query points: %llu,\t ", querypoints);
	printf("K: %d,\t", k);
	std::chrono::duration<double> elapsed = finish - start;
	printf("Elapsed time: %f sec\t", elapsed.count());
	printf("Estimate Mean time: %f sec\n", elapsed.count() / testrepetitions);
	auto timenowend = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	if (!onelineoutpout)
		printf("Experiment Ended: %s\n", ctime(&timenowend));

	


	return 0;
}
