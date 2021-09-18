#pragma once

#ifdef RTX_EXPORTS
	#ifdef _MSC_VER
		#define DLL_API __declspec(dllexport)
	#elif(__GNUG__)
		#if __clang__
			#define DLL_API __attribute__((visibility("default")))
		#else
			#define DLL_API __attribute__((visibility("default")))
		#endif
	#else
		#define DLL_API
	#endif
#else
	#ifdef _MSC_VER
		#define DLL_API __declspec(dllimport)
	#else
		#define DLL_API
	#endif //_MSC_VER
#endif

extern "C" {

	/// Initialize OptiX and CUDA and prepare a context ready for device 0
	/// @return 0 on success and appropriate error code on error
	DLL_API int initRTX();

	/// Return the hash value that identifies the current acceleration structure 
	/// @return uint64(-1) if no acceleration structure is present and the hash
	/// that was provided to buildRTX if it is present
	DLL_API uint64_t getHashRTX();

	/// Build an OptiX ray/triangle acceleration structure based on the vertex and index buffer provided
	/// @param hash A hash value that should uniquely identify the index and vertex buffers
	/// @param verts Pointer to the vertex buffer. Format is 3 float32 per vertex.
	/// @param vBytes The size of the vertex buffer in bytes.
	/// @param triangles Pointer to the index buffer. Format is 3 int32 per triangle.
	/// @param tBytes The size of the index buffer in bytes.
	/// @return 0 on success and appropriate error code on error
	DLL_API int buildRTX(uint64_t hash, void* verts, int64_t vBytes, void* triangles, int tBytes);

	/// Trace the array of rays provided and return the result in hits
	/// @param rays A buffer of rays to trace. Each ray is made of 8 float32 and has the following layout:
	///             [0:3] - ray origin
	///             [3] - ray minT (intersection closer than this value are ignored)
	///             [4:7] - ray direction (normalized)
	///             [7] - ray maxT (intersections farther than this value are ignored)
	/// @param hits A buffer containing data for each hit. Each entry is made of 4 float32 and they have the following layout:
	///             [0] - distance to the intersection point or -1 if the ray failed to intersect anything
	///             [1:4] - the normal vector at the point of intersection. The vector is of unit length. If there is no valid hit, 
	///             the contents are undefined
	/// @param size The number of rays we're tracing.
	/// @return 0 on success and appropriate error code on error
	DLL_API int traceRTX(void* rays, void* hits, int size);

	/// Deallocate all resources currently owned
	/// @return 0 on success and appropriate error code on error
	DLL_API int cleanRTX();
}
