#ifndef __CUDA_RENDERER_H__
#define __CUDA_RENDERER_H__

#ifndef uint
#define uint unsigned int
#endif

#include "circleRenderer.h"

struct Box {
    float boxL;
    float boxR;
    float boxT;
    float boxB;
};

class CudaRenderer : public CircleRenderer {

private:

    Image* image;
    SceneName sceneName;

    int numberOfCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    float* cudaDevicePosition;
    float* cudaDeviceVelocity;
    float* cudaDeviceColor;
    float* cudaDeviceRadius;
    float* cudaDeviceImageData;

    // TODO: check if everything is initialized correctly in the constructor
    int *cudaDeviceIndexOrderMap;  // compressed per-pixel circle indices
    int *cudaDeviceCounts;         // per-pixel counts
    int *d_orderMap;
    int  numCirclesPadded;
    int  numPixels;

    int imageWidth;
    int imageHeight;

    int tileSize = 128;
    int tilesPerWidth;
    int tilesPerHeight;
    int numTiles;

public:

    CudaRenderer();
    virtual ~CudaRenderer();

    const Image* getImage();

    void setup();

    void loadScene(SceneName name);

    void allocOutputImage(int width, int height);

    void clearImage();

    void advanceAnimation();

    void chunkedPrefixSum(int*, int, int);

    void debugPrintOrderMapTile(const int* d_orderMap,
                                int tile,
                                int numCircles,
                                int numCirclesPadded,
                                int maxColsToPrint /* e.g., 128 or numCircles */);

    void debugPrintCompressedTile(const int* d_indexOrderMap,
                                            const int* d_counts,
                                            int tile,
                                            int maxToPrint /* e.g., 64 */);

    void debugPrintCompressedAll(const int* d_indexOrderMap,
                                           const int* d_counts,
                                           int maxToPrintPerTile /* e.g., 64 */);

    void render();

    void shadePixel(
        float pixelCenterX, float pixelCenterY,
        float px, float py, float pz,
        float* pixelData, 
        int circleIndex);
};


#endif
