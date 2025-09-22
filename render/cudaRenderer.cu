#include <string>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <vector>



#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

#define BLOCK_SIZE 1024

// Number of threads in a block
#define SCAN_BLOCK_DIM BLOCK_SIZE

#include "exclusiveScan.cu_inl"
#include "circleBoxTest.cu_inl"


#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#else
#define cudaCheckError(ans) ans
#endif

////////////////////////////////////////////////////////////////////////////////////////
// All cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

// This stores the global constants
struct GlobalConstants {

    SceneName sceneName;

    int numberOfCircles;
    int numCirclesPadded;

    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;

    int tileSize;
    int tilesPerWidth;
    int tilesPerHeight;
    int numTiles;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// Read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// Color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// Include parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"

// Utility functions

// Helper function to round up to a power of 2.
__host__ __device__
int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__device__
Box getBoundingBoxOfTile(int tileIdx) {

    float invWidth = 1.f / cuConstRendererParams.imageWidth;
    float invHeight = 1.f / cuConstRendererParams.imageHeight;

    int tileX = tileIdx % cuConstRendererParams.tilesPerWidth;
    int tileY = tileIdx / cuConstRendererParams.tilesPerWidth;
    int pixL = tileX * cuConstRendererParams.tileSize;
    int pixR = min((tileX + 1) * cuConstRendererParams.tileSize, cuConstRendererParams.imageWidth) - 1;
    int pixT = min((tileY + 1) * cuConstRendererParams.tileSize, cuConstRendererParams.imageHeight) - 1;
    int pixB =  tileY * cuConstRendererParams.tileSize;

    Box box = {
        .boxL = invWidth * (static_cast<float>(pixL) + 0.5f),
        .boxR = invWidth * (static_cast<float>(pixR) + 0.5f),
        .boxT = invHeight * (static_cast<float>(pixT) + 0.5f),
        .boxB = invHeight * (static_cast<float>(pixB) + 0.5f)
    };

    return box;
}


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // Write to global memory: As an optimization, this code uses a float4
    // store, which results in more efficient code than if it were coded as
    // four separate float stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // Write to global memory: As an optimization, this code uses a float4
    // store, which results in more efficient code than if it were coded as
    // four separate float stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update positions of fireworks
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = M_PI;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // Determine the firework center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // Update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // Firework sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // Compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // Compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // Random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // Travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numberOfCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // Place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the position of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numberOfCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// Move the snowflake animation forward one time step.  Update circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // Load from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // Hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // Add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // Drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // Update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // Update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // If the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // Restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // Store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// Given a pixel and a circle, determine the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(float2 pixelCenter, float3 p, float4* imagePtr, int circleIndex) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // Circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // There is a non-zero contribution.  Now compute the shading value

    // Suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks, etc., to implement the conditional.  It
    // would be wise to perform this logic outside of the loops in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // Simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // Global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
__global__ void kernelRenderCircles() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    int index3 = 3 * index;

    // Read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[index];

    // Compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // A bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // For all pixels in the bounding box
    for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
        float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
        for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
            float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
            shadePixel(pixelCenterNorm, p, imgPtr, index);
            imgPtr++;
        }
    }
}

// Helper function that tells us if a pixel center is inside
// a circle at position p and circleIndex.
__device__ bool isInCircle(float2 pixelCenter, float3 p, int circleIndex) {
    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    if (pixelDist > maxDist)
        return false;
    else
        return true;
}

// Flattens a 3d index.
__device__ int get3dIdx(int x, int y, int z, int xDim, int yDim, int zDim) {
    return z + y * zDim + x * (yDim * zDim);
}

// Orders circles.
__global__ void kernelOrderCircles(int* orderMap, int rowLengthPadded, int numTiles) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= cuConstRendererParams.numberOfCircles)
        return;

    int index3 = 3 * gid;

    // Read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[gid];

    // Iterate through all tiles and see which ones are intersecting
    for (int i = 0; i < numTiles; i++){
        Box bb = getBoundingBoxOfTile(i);
        // do fast check first
        if (!circleInBoxConservative(p.x, p.y, rad, bb.boxL, bb.boxR, bb.boxT, bb.boxB)) {
            continue;
        }
        // Now do actual check
        if (!circleInBox(p.x, p.y, rad, bb.boxL, bb.boxR, bb.boxT, bb.boxB)) {
            continue;
        }
        // If we're here, this means this circle contributes to this tile.
        // For Chris: gid = circle ID
        orderMap[i * rowLengthPadded + gid] = 1;
    }
}

__global__ void kernelScanChunks(const int* d_in,
                                 int* d_out,
                                 int* d_chunkTotals,
                                 int rowLength,
                                 int rowLengthPadded,
                                 int numChunksPerTile) {
    int tile = blockIdx.y;      // tile index
    int chunk = blockIdx.x;      // chunk index within this tile
    int tid   = threadIdx.x;     // index within this chunk

    int base = tile * rowLengthPadded + chunk * blockDim.x;

    extern __shared__ int sMem[];
    int* sInput   = sMem;
    int* sOutput  = sInput + blockDim.x;
    int* sScratch = sOutput + blockDim.x;

    // Guard against overflow if last chunk < blockDim.x
    int endOfRow = tile * rowLengthPadded + rowLength;
    int val = 0;
    if (base + tid < endOfRow) {
        val = d_in[base + tid];
    }
    sInput[tid] = val;
    __syncthreads();

    // Run shared memory scan
    sharedMemExclusiveScan(tid,
                           (uint*)sInput,
                           (uint*)sOutput,
                           (uint*)sScratch,
                           blockDim.x);
    __syncthreads();

    // Write back partial scan
    if (base + tid < tile * rowLengthPadded + rowLength) {
        d_out[base + tid] = sOutput[tid];
    }

    // Thread at end of chunk writes the chunk total
    if (tid == blockDim.x - 1) {
        int total = sOutput[tid] + sInput[tid];
        d_chunkTotals[tile * numChunksPerTile + chunk] = total;
    }
}

__global__ void kernelScanChunkTotals(int* d_chunkTotals,
                                      int numChunksPerTile) {
    int tile = blockIdx.x;
    int tid   = threadIdx.x;

    extern __shared__ int sMem[];
    int* sInput   = sMem;
    int* sOutput  = sInput + blockDim.x;
    int* sScratch = sOutput + blockDim.x; // needs 2*blockDim.x

    // Load/pad: fill first numChunksPerTile values, rest 0
    int val = 0;
    if (tid < numChunksPerTile) {
        val = d_chunkTotals[tile * numChunksPerTile + tid];
    }
    sInput[tid] = val;
    __syncthreads();

    // Scan BLOCK_SIZE lanes (matches SCAN_BLOCK_DIM)
    sharedMemExclusiveScan(tid,
                           (uint*)sInput,
                           (uint*)sOutput,
                           (uint*)sScratch,
                           blockDim.x);
    __syncthreads();

    // Store only the valid part
    if (tid < numChunksPerTile) {
        d_chunkTotals[tile * numChunksPerTile + tid] = sOutput[tid];
    }
}

__global__ void kernelAddOffsets(int* d_out,
                                 const int* d_chunkOffsets,
                                 int rowLength,
                                 int rowLengthPadded,
                                 int numChunksPerTile) {
    int tile = blockIdx.y;
    int chunk = blockIdx.x;
    int tid   = threadIdx.x;

    int base = tile * rowLengthPadded + chunk * blockDim.x;
    int offset = d_chunkOffsets[tile * numChunksPerTile + chunk];

    if (base + tid < tile * rowLengthPadded + rowLength) {
        d_out[base + tid] += offset;
    }
}

// prefix:  [numTiles x ROW_PADDED] exclusive scan, per tile, over ROW_LEN columns
// outIdx:  [numTiles x numberOfCircles] packed circle indices per tile
// counts:  [numTiles] total kept per tile (read from sentinel = prefix[base + numCircles])
__global__ void kernelCompressIndices(const int* __restrict__ prefix,
                                           int* __restrict__ outIdx,
                                           int* __restrict__ counts,
                                           int numberOfCircles,
                                           int rowLengthPadded)
{
    int tile   = blockIdx.y;                                    // 0..numTiles-1
    int circle = blockIdx.x * blockDim.x + threadIdx.x;         // 0..numberOfCircles-1
    if (tile >= cuConstRendererParams.numTiles || circle >= numberOfCircles) return;

    const int base = tile * rowLengthPadded;

    // Exclusive-scan slots:
    //   prefix[base + i]     = write position if flag[i] == 1
    //   prefix[base + i + 1] = next position; different => flag[i]==1
    int curr = prefix[base + circle];
    int next = prefix[base + circle + 1];

    // If the scan advanced, keep this index at position = curr
    if (curr != next) {
        outIdx[tile * numberOfCircles + curr] = circle;
    }

    // One lane writes total count from the sentinel
    if (circle == numberOfCircles - 1) {
        counts[tile] = prefix[base + numberOfCircles];  // sentinel
    }
}

__global__ void kernelRenderTiles(const int* __restrict__ indexOrderMap,
                                                const int* __restrict__ counts,
                                                int numCircles)
{
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;
    int tile  = tileY * cuConstRendererParams.tilesPerWidth + tileX;
    if (tile >= cuConstRendererParams.numTiles) return;

    // Calculate pixels corresponding to this tile
    const int tileSize      = cuConstRendererParams.tileSize;
    const int imgW          = cuConstRendererParams.imageWidth;
    const int imgH          = cuConstRendererParams.imageHeight;

    const int startX = tileX * tileSize;
    const int startY = tileY * tileSize;
    const int endX   = min(startX + tileSize, imgW); // exclusive
    const int endY   = min(startY + tileSize, imgH); // exclusive

    // Float reciprocals for pixel normalization
    float invWidth = 1.f / imgW;
    float invHeight = 1.f / imgH;

    // Loop over pixels with striding
    for (int py = startY + threadIdx.y; py < endY; py += blockDim.y) {
        for (int px = startX + threadIdx.x; px < endX; px += blockDim.x) {
            // Process blockDim.x by blockDim.y window in current tile.

            // Normalized pixel center
            // Y = 0 is the bottom of the image
            float2 pixelCenter = make_float2(
                invWidth * (px + 0.5f),
                invHeight * (py + 0.5f)
            );

            // Pointer to pixel RGBA
            float4* imagePtr = (float4*)(&cuConstRendererParams.imageData[4 * (py * imgW + px)]);

            // How many circles overlap this tile? If zero, we can skip.
            const int count = counts[tile];
            if (count <= 0) continue;

            // Base of this tile's circle indices.
            const int base = tile * numCircles;

            // Accumulate this pixel over relevant circles
            for (int k = 0; k < count; ++k) {
                const int circleIdx = indexOrderMap[base + k];

                // Load circle center (xyz) and shade; shadePixel will early-out if outside radius
                const float3 p = *(const float3*)(&cuConstRendererParams.position[3 * circleIdx]);
                shadePixel(pixelCenter, p, imagePtr, circleIdx);
            }
        }
    }
}


////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numberOfCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;

    cudaDeviceIndexOrderMap = NULL;
    cudaDeviceCounts = NULL;
    numCirclesPadded = nextPow2(numberOfCircles);
    numPixels = 0;

    imageWidth = 0;
    imageHeight = 0;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image*
CudaRenderer::getImage() {

    // Need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numberOfCircles, position, velocity, color, radius);
}

void CudaRenderer::chunkedPrefixSum(int* d_orderMap, int rowLength, int rowLengthPadded) {
    int numChunksPerTile = (rowLengthPadded + SCAN_BLOCK_DIM - 1) / SCAN_BLOCK_DIM;

    // Device buffers
    int* d_chunkTotals;
    cudaCheckError(cudaMalloc(&d_chunkTotals, sizeof(int) * numTiles * numChunksPerTile));

    // Pass 1: scan each chunk
    dim3 grid1(numChunksPerTile, numTiles);
    size_t sharedMemSize = (4 * SCAN_BLOCK_DIM) * sizeof(int);
    kernelScanChunks<<<grid1, SCAN_BLOCK_DIM, sharedMemSize>>>(d_orderMap,
                                                               d_orderMap,
                                                               d_chunkTotals,
                                                               rowLength,
                                                               rowLengthPadded,
                                                               numChunksPerTile);
    cudaCheckError(cudaDeviceSynchronize());

    // Pass 2: scan chunk totals
    kernelScanChunkTotals<<<numTiles, numChunksPerTile, sharedMemSize>>>(d_chunkTotals, numChunksPerTile);
    cudaCheckError(cudaDeviceSynchronize());

    // Pass 3: add offsets back
    dim3 grid3(numChunksPerTile, numTiles);
    kernelAddOffsets<<<grid3, SCAN_BLOCK_DIM>>>(
        d_orderMap, d_chunkTotals,
        rowLength, rowLengthPadded, numChunksPerTile);
    cudaCheckError(cudaDeviceSynchronize());

    cudaCheckError(cudaFree(d_chunkTotals));
}


void
CudaRenderer::setup() {

    int deviceCount = 0;
    bool isFastGPU = false;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;
        if (name.compare("GeForce RTX 2080") == 0)
        {
            isFastGPU = true;
        }

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    if (!isFastGPU)
    {
        printf("WARNING: "
               "You're not running on a fast GPU, please consider using "
               "NVIDIA RTX 2080.\n");
        printf("---------------------------------------------------------\n");
    }
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numberOfCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numberOfCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    imageWidth = image->width;
    imageHeight = image->height;
    tilesPerWidth = (image->width + tileSize - 1) / tileSize;
    tilesPerHeight = (image->height + tileSize - 1) / tileSize;
    numTiles = tilesPerHeight * tilesPerWidth;
    numCirclesPadded = nextPow2(numberOfCircles);

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numberOfCircles = numberOfCircles;
    params.numCirclesPadded = numCirclesPadded;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;
    params.tilesPerHeight = tilesPerHeight;
    params.tilesPerWidth = tilesPerWidth;
    params.tileSize = tileSize;
    params.numTiles = numTiles;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // Also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // Copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numberOfCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}

// void incorrectRender() {
//     // 256 threads per block is a healthy number
//     dim3 blockDim(256, 1);
//     dim3 gridDim((numberOfCircles + blockDim.x - 1) / blockDim.x);

//     kernelRenderCircles<<<gridDim, blockDim>>>();
//     cudaDeviceSynchronize();
// }

/* 
 * PRETTY PRINTING FUNCTIONS 
**/

void CudaRenderer::debugPrintOrderMapTile(const int* d_orderMap,
                                          int tile,
                                          int numCircles,
                                          int rowLengthPadded,
                                          int maxColsToPrint /* e.g., 128 or numCircles */)
{
    if (tile < 0 || tile >= numTiles) {
        printf("Tile %d out of range [0, %d)\n", tile, numTiles);
        return;
    }

    cudaCheckError(cudaDeviceSynchronize());

    std::vector<int> h_row(rowLengthPadded);
    const size_t rowBytes = static_cast<size_t>(rowLengthPadded) * sizeof(int);
    cudaCheckError(cudaMemcpy(h_row.data(),
                              d_orderMap + tile * rowLengthPadded,
                              rowBytes,
                              cudaMemcpyDeviceToHost));

    int tileX = tile % tilesPerWidth;
    int tileY = tile / tilesPerWidth;
    printf("==== Tile %d (%d,%d) Row ====\n", tile, tileX, tileY);

    int colsToPrint = std::min(numCircles, maxColsToPrint);
    // print grouped for readability
    for (int i = 0; i < colsToPrint; i += 32) {
        int end = std::min(i + 32, colsToPrint);
        printf("[%4d..%4d): ", i, end);
        for (int c = i; c < end; ++c) {
            printf("%d", h_row[c]);
        }
        printf("\n");
    }

    if (colsToPrint < numCircles) {
        printf("... (%d more columns not shown)\n", numCircles - colsToPrint);
    }
    // also show tail of padded region if you care
    // printf("Padded tail (last 16): ");
    // for (int c = rowLengthPadded - 16; c < rowLengthPadded; ++c) printf("%d", h_row[c]);
    // printf("\n");

    printf("=============================\n");
}

// Pretty-print the compressed index list for a single tile.
// d_indexOrderMap layout: [numTiles][numberOfCircles] row-major by tile
// d_counts layout:        [numTiles] (how many valid indices per tile)
void CudaRenderer::debugPrintCompressedTile(const int* d_indexOrderMap,
                                            const int* d_counts,
                                            int tile,
                                            int maxToPrint /* e.g., 64 */)
{
    if (tile < 0 || tile >= numTiles) {
        printf("Tile %d out of range [0, %d)\n", tile, numTiles);
        return;
    }

    // sync before reading
    cudaCheckError(cudaDeviceSynchronize());

    // pull count
    int h_count = 0;
    cudaCheckError(cudaMemcpy(&h_count,
                              d_counts + tile,
                              sizeof(int),
                              cudaMemcpyDeviceToHost));
    h_count = std::max(0, std::min(h_count, numberOfCircles));

    // nothing in this tile?
    int tileX = tile % tilesPerWidth;
    int tileY = tile / tilesPerWidth;
    printf("==== Compressed indices for Tile %d (%d,%d) ====\n", tile, tileX, tileY);
    printf("count = %d\n", h_count);
    if (h_count == 0) {
        printf("(empty)\n");
        printf("===============================================\n");
        return;
    }

    // pull the first min(h_count, maxToPrint) entries
    const int toCopy = std::min(h_count, maxToPrint);
    std::vector<int> h_idx(toCopy);
    const size_t rowStride = static_cast<size_t>(numberOfCircles);
    cudaCheckError(cudaMemcpy(h_idx.data(),
                              d_indexOrderMap + tile * rowStride,
                              sizeof(int) * toCopy,
                              cudaMemcpyDeviceToHost));

    // print in small groups for readability
    for (int i = 0; i < toCopy; i += 16) {
        int end = std::min(i + 16, toCopy);
        printf("[%4d..%4d): ", i, end);
        for (int j = i; j < end; ++j) {
            printf("%d ", h_idx[j]);
        }
        printf("\n");
    }
    if (toCopy < h_count) {
        printf("... (%d more indices not shown)\n", h_count - toCopy);
    }
    printf("===============================================\n");
}

// Convenience: print several (or all) tiles
void CudaRenderer::debugPrintCompressedAll(const int* d_indexOrderMap,
                                           const int* d_counts,
                                           int maxToPrintPerTile /* e.g., 64 */)
{
    for (int t = 0; t < numTiles; ++t) {
        debugPrintCompressedTile(d_indexOrderMap, d_counts, t, maxToPrintPerTile);
    }
}


/* 
 * PRETTY PRINTING FUNCTIONS END
**/

void CudaRenderer::render() {
    const int ROW_LEN = numberOfCircles + 1; // Need a sentinel value for exclusive prefix sum
    const int ROW_PADDED = nextPow2(ROW_LEN);
    int *d_orderMap;
    cudaCheckError(cudaMalloc(&d_orderMap, sizeof(int) * numTiles * ROW_PADDED));
    cudaCheckError(cudaMemset(d_orderMap, 0, sizeof(int) * numTiles * ROW_PADDED));

    // Initialize order map
    int threadsPerBlock = BLOCK_SIZE;
    int numBlocks = (ROW_PADDED + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernelOrderCircles<<<numBlocks, threadsPerBlock>>>(d_orderMap, ROW_PADDED, numTiles);
    cudaCheckError(cudaDeviceSynchronize());

    // Get EPS using ES with chunking
    chunkedPrefixSum(d_orderMap, ROW_LEN, ROW_PADDED);



    // Compress order bit vectors into concrete circle indices
    cudaCheckError(cudaMalloc(&cudaDeviceIndexOrderMap, sizeof(int) * numTiles * numberOfCircles));
    cudaCheckError(cudaMalloc(&cudaDeviceCounts, sizeof(int) * numTiles));
    // Memset to -1 and 0 each respectively.
    cudaCheckError(cudaMemset(cudaDeviceIndexOrderMap, 0xFF,
                          sizeof(int) * numTiles * numberOfCircles));
    cudaCheckError(cudaMemset(cudaDeviceCounts, 0, sizeof(int) * numTiles));

    // // Choose a reasonable tile for circles (no need for 1024 here)
    dim3 block(BLOCK_SIZE);
    dim3 grid((numberOfCircles + BLOCK_SIZE - 1) / BLOCK_SIZE,  // over circles
            numTiles);                                            // over tiles
              // over pixels

    kernelCompressIndices<<<grid, block>>>(d_orderMap,
                                            cudaDeviceIndexOrderMap,
                                            cudaDeviceCounts,
                                            /*numberOfCircles=*/numberOfCircles,
                                            /*rowLengthPadded=*/ROW_PADDED);
    

    dim3 blockTile(16, 16); // Block size is still 256 here, just 2d for easier math.
    dim3 gridTiles(tilesPerWidth, tilesPerHeight); // We deploy a block per tile.

    kernelRenderTiles<<<gridTiles, blockTile>>>(cudaDeviceIndexOrderMap,
                                                cudaDeviceCounts,
                                                numberOfCircles);
    cudaFree(d_orderMap);
    cudaCheckError(cudaDeviceSynchronize());
}