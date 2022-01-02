#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <jni.h>
#include <android/log.h>
#include <android/bitmap.h>

#include "CL/opencl.h"

#define LOG_TAG "POE"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG,__VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)

#define checkCL(expression) {                        \
	cl_int err = (expression);                       \
	if (err < 0 && err > -64) {                      \
		LOGE("POE: Error on line %d. error code: %d\n", \
				__LINE__, err);                      \
		exit(0);                                     \
	}                                                \
}                                                    \

#include "NeuralNet.cpp"

int opencl_infra_creation(cl_context&       context,
                          cl_platform_id&   cpPlatform,
                          cl_device_id&     device_id,
                          cl_command_queue& queue,
                          cl_program&       program,
                          cl_kernel&        kernel_t,
                          cl_kernel&        kernel_d,
                          cl_kernel&        kernel_e,
                          char*             kernel_file_buffer,
                          size_t            kernel_file_size,
                          unsigned char*    kernel_name_t,
                          unsigned char*    kernel_name_d,
                          unsigned char*    kernel_name_e,
                          cl_mem&           d_src,
                          cl_mem&           d_dst
) {

    cl_int err;

    // Bind to platform
    checkCL(clGetPlatformIDs(1, &cpPlatform, NULL));

    // Get ID for the device
    checkCL(clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL));

    // Create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

    // Create a command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    checkCL(err);

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1,
                                        (const char **) & kernel_file_buffer, &kernel_file_size, &err);
    checkCL(err);

    // Build the program executable
    checkCL(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));

    // Create the compute kernel in the program we wish to run
    kernel_t = clCreateKernel(program, (const char*)kernel_name_t, &err);
    checkCL(err);
    kernel_d = clCreateKernel(program, (const char*)kernel_name_d, &err);
    checkCL(err);
    kernel_e = clCreateKernel(program, (const char*)kernel_name_e, &err);
    checkCL(err);

    free(kernel_file_buffer);

    return 0;
}

int launch_the_kernel_AdaptiveThreshold(cl_context&       context,
                                        cl_command_queue& queue,
                                        cl_kernel&        kernel,
                                        size_t            globalSize,
                                        size_t            localSize,
                                        int height, int width,
                                        cl_mem&           d_src,
                                        cl_mem&           d_dst,
                                        unsigned char*    image,
                                        unsigned char*    blured_img,
                                        int adp_size, int threshold
) {

    cl_int err;

    // Create the input and output arrays in device memory for our calculation
    //d_src = clCreateBuffer(context, CL_MEM_READ_ONLY, (height * width * 4), NULL, &err);
    //checkCL(err);
    //d_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (height * width * 4), NULL, &err);
    //checkCL(err);

    // Write our data set into the input array in device memory
    checkCL(clEnqueueWriteBuffer(queue, d_src, CL_TRUE, 0,
                                 height * width*4 , image, 0, NULL, NULL));

    // Set the arguments to our compute kernel
    checkCL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_src));
    checkCL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_dst));
    checkCL(clSetKernelArg(kernel, 2, sizeof(int), &width));
    checkCL(clSetKernelArg(kernel, 3, sizeof(int), &height));
    checkCL(clSetKernelArg(kernel, 4, sizeof(int), &adp_size));
    checkCL(clSetKernelArg(kernel, 5, sizeof(int), &threshold));

    // Execute the kernel over the entire range of the data set
    checkCL(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                                   0, NULL, NULL));
    // Wait for the command queue to get serviced before reading back results
    checkCL(clFinish(queue));

    // Read the results from the device
    checkCL(clEnqueueReadBuffer(queue, d_dst, CL_TRUE, 0,
                                (height * width * 4 * sizeof(unsigned char)), blured_img, 0, NULL, NULL ));

    return 0;
}

int launch_the_kernel_Dilation(cl_context&       context,
                               cl_command_queue& queue,
                               cl_kernel&        kernel,
                               size_t            globalSize,
                               size_t            localSize,
                               int height, int width,
                               cl_mem&           d_src,
                               cl_mem&           d_dst,
                               unsigned char*    image,
                               unsigned char*    blured_img,
                               int bold_size
) {

    cl_int err;
    struct timeval start, end, timer;

    // Create the input and output arrays in device memory for our calculation
    //d_src = clCreateBuffer(context, CL_MEM_READ_ONLY, (height * width * 4), NULL, &err);
    //checkCL(err);
    //d_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (height * width * 4), NULL, &err);
    //checkCL(err);

    // Write our data set into the input array in device memory
    checkCL(clEnqueueWriteBuffer(queue, d_src, CL_TRUE, 0,
                                 (height * width * 4), image, 0, NULL, NULL));

    // Set the arguments to our compute kernel
    checkCL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_src));
    checkCL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_dst));
    checkCL(clSetKernelArg(kernel, 2, sizeof(int), &width));
    checkCL(clSetKernelArg(kernel, 3, sizeof(int), &height));
    checkCL(clSetKernelArg(kernel, 4, sizeof(int), &bold_size));

    // Execute the kernel over the entire range of the data set
    checkCL(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                                   0, NULL, NULL));
    // Wait for the command queue to get serviced before reading back results
    checkCL(clFinish(queue));

    // Read the results from the device
    checkCL(clEnqueueReadBuffer(queue, d_dst, CL_TRUE, 0,
                                (height * width * 4), blured_img, 0, NULL, NULL ));

    return 0;
}

int launch_the_kernel_Erosion(cl_context&       context,
                              cl_command_queue& queue,
                              cl_kernel&        kernel,
                              size_t            globalSize,
                              size_t            localSize,
                              int height, int width,
                              cl_mem&           d_src,
                              cl_mem&           d_dst,
                              unsigned char*    image,
                              unsigned char*    blured_img,
                              int thin_size
) {

    cl_int err;
    struct timeval start, end, timer;

    // Create the input and output arrays in device memory for our calculation
    //d_src = clCreateBuffer(context, CL_MEM_READ_ONLY, (height * width * 4), NULL, &err);
    //checkCL(err);
    //d_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (height * width * 4), NULL, &err);
    //checkCL(err);

    // Write our data set into the input array in device memory
    checkCL(clEnqueueWriteBuffer(queue, d_src, CL_TRUE, 0,
                                 (height * width * 4), image, 0, NULL, NULL));

    // Set the arguments to our compute kernel
    checkCL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_src));
    checkCL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_dst));
    checkCL(clSetKernelArg(kernel, 2, sizeof(int), &width));
    checkCL(clSetKernelArg(kernel, 3, sizeof(int), &height));
    checkCL(clSetKernelArg(kernel, 4, sizeof(int), &thin_size));

    // Execute the kernel over the entire range of the data set
    checkCL(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                                   0, NULL, NULL));
    // Wait for the command queue to get serviced before reading back results
    checkCL(clFinish(queue));

    // Read the results from the device
    checkCL(clEnqueueReadBuffer(queue, d_dst, CL_TRUE, 0,
                                (height * width * 4), blured_img, 0, NULL, NULL ));

    return 0;
}
//OpenCL


typedef struct Rectangle {
    int sx;
    int sy;
    int ex;
    int ey;
} Contour;

Contour BFS(unsigned char* img, int x, int y, int width, int height){
    Contour c;
    int kx, ky;
    int queue[2000][2];
    int front, end;
    c.sx = x; c.ex = x; c.sy = y; c.ey = y;
    front = 0; end = 0;
    queue[end][0] = x; queue[end][1] = y; end=(end+1)%2000;
    while(front!=end){
        kx = queue[front][0]; ky = queue[front][1]; front=(front+1)%2000;
        if(0>kx || 0>ky || kx>=width || ky>=height) continue;
        if(img[(ky*width+kx)]!=255) continue;
        if(c.sx>kx) c.sx=kx;
        if(c.sy>ky) c.sy=ky;
        if(c.ex<kx) c.ex=kx;
        if(c.ey<ky) c.ey=ky;
        img[(ky*width+kx)] = 2;
        queue[end][0] = kx-1; queue[end][1] = ky; end=(end+1)%2000;
        queue[end][0] = kx+1; queue[end][1] = ky; end=(end+1)%2000;
        queue[end][0] = kx; queue[end][1] = ky-1; end=(end+1)%2000;
        queue[end][0] = kx; queue[end][1] = ky+1; end=(end+1)%2000;
    }
    return c;
}
//입력: 출력될 contour array, img, 검출넓이
//출력: contour 개수
int detectContours(Contour* c, unsigned char* img, int width, int height, int thresholdArea, int margin){
    int i, j, n;
    Contour tc;
    unsigned char* t_img = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    for(i=0;i<width*height;i++) t_img[i] = img[i*4];
    n = 0;
    for(i=0;i<height;i++){
        for(j=0;j<width;j++){
            if(t_img[i*width+j]==255){
                tc = BFS(t_img, j, i, width, height);
                if((tc.ex-tc.sx)*(tc.ey-tc.sy)>thresholdArea && ((tc.ex-tc.sx)>thresholdArea || (tc.ey-tc.sy)>thresholdArea)){
                    tc.sx-=margin; tc.sy-=margin;
                    tc.ex+=margin; tc.ey+=margin;
                    c[n++] = tc;
                }
            }
        }
    }
    free(t_img);
    return n;
}
//contour sorting: pibot=sx, rising
void contour_sort(Contour* c, int n){
    int i, j;
    Contour tmp;
    for(i=0;i<n;i++){
        for(j=i+1;j<n;j++){
            if(c[i].sx > c[j].sx){
                tmp = c[i];
                c[i] = c[j];
                c[j] = tmp;
            }
        }
    }
}

void near_resize(unsigned char *data, unsigned char *dst, int width, int height, int newWidth, int newHeight)
{
    int i,j;
    int x,y;
    float tx,ty;

    tx = (float)width /newWidth ;
    ty =  (float)height / newHeight;

    for(i=0; i<newHeight; i++)
        for(j=0; j<newWidth; j++)
        {
            x = (int)j*tx;
            y = (int)i*ty;
            dst[i * newWidth + j] = data[y*width+x];
        }
}

double dist(double a, double b, double c, double d){
    return sqrt(pow(a-c,2)+pow(b-d,2));
}



extern "C"
JNIEXPORT void JNICALL
Java_com_example_uos_project_JNIProcess_init(JNIEnv *env, jclass type, jint width, jint height) {

}

extern "C"
JNIEXPORT jintArray JNICALL
Java_com_example_uos_project_JNIProcess_process(JNIEnv *env, jclass type, jint width, jint height,
                                                jobject bitmap, jbyteArray pixels_, jbyteArray bold_pixels_, jintArray t_nums_, jint nums_n, jintArray nums_group_, jint nums_group_n,
                                                jintArray t_devs_, jint devs_n, jint threshold) {
    //jbyte *NV21FrameData = env->GetByteArrayElements(NV21FrameData_, NULL);
    jbyte *ipixels = env->GetByteArrayElements(pixels_, NULL);
    void* pixels = (void*)ipixels;
    jbyte *ibold_pixels = env->GetByteArrayElements(bold_pixels_, NULL);
    unsigned char* bold_pixels = (unsigned char*)ibold_pixels;
    jint *it_nums = env->GetIntArrayElements(t_nums_, NULL);
    int* t_nums = (int*)it_nums;
    jint *inums_group = env->GetIntArrayElements(nums_group_, NULL);
    int* nums_group = (int*)inums_group;
    jint *it_devs = env->GetIntArrayElements(t_devs_, NULL);
    int* t_devs = (int*)it_devs;
    jintArray returns = env->NewIntArray(3);
    jint* ns = env->GetIntArrayElements(returns, NULL);
    //unsigned char* tempPixels = (unsigned char*)malloc(sizeof(unsigned char) * 4 * width * height);


    LOGD("POE: start");
    //nv21_to_argb(pixels, (unsigned char*)NV21FrameData, width, height);
    //memcpy(pixels, NV21FrameData, width * height * 4);
    LOGD("reading bitmap info..");
    AndroidBitmapInfo info;
    int ret;
    if((ret = AndroidBitmap_getInfo(env, bitmap, &info)) < 0){
        LOGE("AndroidBitmap_getInfor failed: %d",ret);
        return NULL;
    }
    LOGD("width:%d height:%d stride:%d", info.width, info.height, info.stride);

    if(info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
    {
        LOGE("Bitmap Format is not RGBA 8888");
        return NULL;
    }

    LOGD("reading Bitmap pixels..");
    void* bitmapPixels;
    if((ret = AndroidBitmap_lockPixels(env, bitmap, &bitmapPixels)) < 0){
        LOGE("AndroidBitmap_lockPixels failed: %d",ret);
        return NULL;
    }


    FILE *file_handle;
    char *kernel_file_buffer, *file_log;
    size_t kernel_file_size, log_size;
    const char* cl_file_name = "/data/local/tmp/ImageProc.cl";
    unsigned char* kernel_name_t = (unsigned  char*)"AdaptiveThreshold";
    unsigned char* kernel_name_d = (unsigned  char*)"Dilation";
    unsigned char* kernel_name_e = (unsigned  char*)"Erosion";

    // Device input buffers
    cl_mem d_src;
    // Device output buffer
    cl_mem d_dst;

    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel_t;                 // kernel
    cl_kernel kernel_d;                 // kernel
    cl_kernel kernel_e;                 // kernel

    file_handle=fopen(cl_file_name, "r");
    if(file_handle==NULL)
    {
        printf("Couldn't find the file");
        exit(1);
    }

    //read kernel file
    fseek(file_handle, 0, SEEK_END);
    kernel_file_size =ftell(file_handle);
    rewind(file_handle);
    kernel_file_buffer = (char*)malloc(sizeof(char)*(kernel_file_size+1));
    kernel_file_buffer[kernel_file_size]='\0';
    fread(kernel_file_buffer, sizeof(char), kernel_file_size, file_handle);
    fclose(file_handle);

    int i, j;
    size_t globalSize, localSize, grid;
    int pixelCount = width*height;

    uint32_t* src = (uint32_t*)bitmapPixels;
    uint32_t* tempPixels = (uint32_t*)malloc(height*width*4);
    memcpy(tempPixels, src, sizeof(uint32_t)*pixelCount);
    // Number of work items in each local work group
    localSize = 64;

    // Number of total work items - localSize must be devisor
    grid = (pixelCount % localSize) ? (pixelCount/localSize)+1 : (pixelCount/localSize);
    globalSize = grid * localSize;

    opencl_infra_creation(context, cpPlatform, device_id, queue, program,
                          kernel_t,kernel_d,kernel_e, kernel_file_buffer, kernel_file_size,
                          kernel_name_t,kernel_name_d,kernel_name_e,
                          d_src, d_dst
    );
// Create the input and output arrays in device memory for our calculation
    cl_int err;
    d_src = clCreateBuffer(context, CL_MEM_READ_ONLY, (height * width * 4), NULL, &err);
    checkCL(err);
    d_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (height * width * 4), NULL, &err);
    checkCL(err);

    LOGD("POE: threshold: %d",threshold);
    launch_the_kernel_AdaptiveThreshold(context, queue, kernel_t, globalSize, localSize, height, width, //adaptive: gray
                                        d_src, d_dst, (unsigned char*)tempPixels, (unsigned char*)pixels /*40, 26*/, threshold, 30);
    //checkCL(clReleaseMemObject(d_src));
    //checkCL(clReleaseMemObject(d_dst));
    launch_the_kernel_Dilation(context, queue, kernel_d, globalSize, localSize, height, width, //bold:2 -> letter dectect
                               d_src, d_dst, (unsigned char*)pixels, (unsigned char*)tempPixels, /*2*/2);
    //checkCL(clReleaseMemObject(d_src));
    //checkCL(clReleaseMemObject(d_dst));

    Contour cont_n[50]; //letter detect
    int cont_n_n = 0;
    cont_n_n = detectContours(cont_n, (unsigned char*)tempPixels, width, height, 10, 0);

    for(int k=0;k<cont_n_n;k++){ //글자 제거
        if((cont_n[k].ex-cont_n[k].sx)*(cont_n[k].ey-cont_n[k].sy)<=2000){
            for(i=cont_n[k].sy;i<cont_n[k].ey;i++){
                for(j=cont_n[k].sx;j<cont_n[k].ex;j++){
                    ((unsigned char*)tempPixels)[(i*width+j)*4] = 0;
                    ((unsigned char*)tempPixels)[(i*width+j)*4+1] = 0;
                    ((unsigned char*)tempPixels)[(i*width+j)*4+2] = 0;
                }
            }
        }
    }
    for(i=0;i<width*height;i++) bold_pixels[i] = ((unsigned char*)tempPixels)[(i*4)];


    launch_the_kernel_Dilation(context, queue, kernel_d, globalSize, localSize, height, width, //bold:6(closing & line remove) -> circuit device dectect
                               d_src, d_dst, (unsigned char*)tempPixels, (unsigned char*)tempPixels, 11);
    launch_the_kernel_Erosion(context, queue, kernel_e, globalSize, localSize, height, width, //thin:10
                              d_src, d_dst, (unsigned char*)tempPixels, (unsigned char*)tempPixels, 15);
    checkCL(clReleaseMemObject(d_src));
    checkCL(clReleaseMemObject(d_dst));

    Contour cont_c[50]; //circuit device detect
    int cont_c_n = 0;
    cont_c_n = detectContours(cont_c, (unsigned char*)tempPixels, width, height, 30, 6);


    //letter, device detect
    Contour nums[50];
    Contour devs[50];
    nums_n = 0;
    devs_n = 0;
    int maximum = 0;
    for(i=0;i<cont_n_n;i++){
        if((cont_n[maximum].ex-cont_n[maximum].sx)*(cont_n[maximum].ey-cont_n[maximum].sy) < (cont_n[i].ex-cont_n[i].sx)*(cont_n[i].ey-cont_n[i].sy)){
            maximum = i;
        }
    }
    for(i=0;i<cont_n_n;i++){
        if(i==maximum) continue;
        for(j=0;j<cont_c_n;j++){
            if(cont_n[i].sx>=cont_c[j].sx && cont_n[i].sy>=cont_c[j].sy && cont_n[i].ex<=cont_c[j].ex && cont_n[i].ey<=cont_c[j].ey){
                break;
            }
        }
        if(j==cont_c_n && cont_n[i].sx>10 && cont_n[i].sy>10 && cont_n[i].ex<width-10 && cont_n[i].ey<height-10) nums[nums_n++] = cont_n[i]; //기호 밖, 사진 테두리 안
    }
    for(i=0;i<cont_c_n;i++){
        devs[devs_n++] = cont_c[i];
    }


    //latter grouping
    nums_group_n = 0;       ///***
    Contour cont;
    contour_sort(nums, nums_n); //sorting by sx

    Contour nums2[50];
    for(i=0;i<nums_n;i++){
        nums2[i].sx = nums[i].sx-(nums[i].ex-nums[i].sx)/2;
        nums2[i].sy = nums[i].sy-(nums[i].ey-nums[i].sy)/2;
        nums2[i].ex = nums[i].ex+(nums[i].ex-nums[i].sx)/2;
        nums2[i].ey = nums[i].ey+(nums[i].ey-nums[i].sy)/2;
        nums_group[i] = 0;
    }
    for(i=0;i<nums_n;i++){
        if(nums_group[i]==0){
            cont = nums2[i];
            nums_group[i] = ++nums_group_n;
            for(j=i+1;j<nums_n;j++){
                if(cont.sx<=nums2[j].ex && cont.ex>=nums2[j].sx && cont.sy<=nums2[j].ey && cont.ey>=nums2[j].sy && nums_group[j]==0){ //overlaped
                    nums_group[j] = nums_group_n;
                    if(cont.sx>nums2[j].sx) cont.sx=nums2[j].sx;
                    if(cont.sy>nums2[j].sy) cont.sy=nums2[j].sy;
                    if(cont.ex<nums2[j].ex) cont.ex=nums2[j].ex;
                    if(cont.ey<nums2[j].ey) cont.ey=nums2[j].ey;
                }
            }
        }
    }

    for(i=0;i<nums_n;i++){
        t_nums[i*4] = nums[i].sx;
        t_nums[i*4+1] = nums[i].sy;
        t_nums[i*4+2] = nums[i].ex;
        t_nums[i*4+3] = nums[i].ey;
    }
    for(i=0;i<devs_n;i++){
        t_devs[i*4] = devs[i].sx;
        t_devs[i*4+1] = devs[i].sy;
        t_devs[i*4+2] = devs[i].ex;
        t_devs[i*4+3] = devs[i].ey;
    }

//    launch_the_kernel_Dilation(context, queue, kernel_d, globalSize, localSize, height, width, //bold:2 -> letter dectect
    //                              d_src, d_dst, (unsigned char*)pixels, (unsigned char*)tempPixels, 6);
    //   dectectNeib((unsigned char*)tempPixels, width, height, circuit_edge, t_devs, devs_n);
    //checkCL(clReleaseMemObject(d_src));
    //checkCL(clReleaseMemObject(d_dst));
    checkCL(clReleaseProgram(program));
    checkCL(clReleaseKernel(kernel_t));
    checkCL(clReleaseKernel(kernel_d));
    checkCL(clReleaseKernel(kernel_e));
    checkCL(clReleaseCommandQueue(queue));
    checkCL(clReleaseContext(context));

    free(tempPixels);

    LOGD("POE: end");
    ns[0] = nums_n;
    ns[1] = nums_group_n;
    ns[2] = devs_n;
    // Release
    AndroidBitmap_unlockPixels(env, bitmap);
    (env)->ReleaseByteArrayElements(pixels_, ipixels, 0);
    (env)->ReleaseByteArrayElements(bold_pixels_, ibold_pixels, 0);
    (env)->ReleaseIntArrayElements(t_nums_, it_nums, 0);
    (env)->ReleaseIntArrayElements(nums_group_, inums_group, 0);
    (env)->ReleaseIntArrayElements(t_devs_, it_devs, 0);
    (env)->ReleaseIntArrayElements(returns, ns, 0);

    return returns;
}

void detmid(unsigned char* piece, int width, int height, int *coor){
    double sumx = 0, maxx = 0, minx = 99999999, sumy = 0, maxy = 0, miny = 99999999;
    double sx[200];
    double sy[200];
    int a, b;
    for(a=0;a<width;a++) sx[a] = 0;
    for(b=0;b<height;b++) sy[b] = 0;
    for(a=0;a<width;a++){
        for(b=0;b<height;b++){
            sx[a] += piece[b*width+a];
            sy[b] += piece[b+width+a];
        }
    }
    for(a=0;a<width;a++){
        sx[a] = pow(sx[a],1.25);
        if(maxx<sx[a]) maxx=sx[a];
        if(minx>sx[a]) minx=sx[a];
    }
    for(b=0;b<height;b++){
        sy[b] = pow(sy[b],1.25);
        if(maxy<sy[b]) maxy=sy[b];
        if(miny>sy[b]) miny=sy[b];
    }
    for(a=width/4;a<width*3/4;a++){
        sx[a] = (sx[a]-minx)/maxx;
        sumx+=sx[a];
    }
    for(b=height/4;b<height*3/4;b++){
        sy[b] = (sy[b]-miny)/maxy;
        sumy+=sy[b];
    }
    double midx = 0, midy = 0;
    for(a=width/4;a<width*3/4;a++){
        midx+=sx[a];
        if(midx>=sumx/2){
            coor[0] = a;
            break;
        }
    }
    for(b=height/4;b<height*3/4;b++){
        midy+=sy[b];
        if(midy>=sumy/2){
            coor[1] = b;
            break;
        }
    }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_uos_project_JNIProcess_matchinglabel(JNIEnv *env, jclass type, jbyteArray pixels_,
                                                      jint width, jint height, jintArray t_nums_,
                                                      jintArray nums_label_, jint nums_n,
                                                      jintArray t_devs_, jintArray devs_label_, jintArray devs_bipolar_,
                                                      jint devs_n) {
    jbyte *ipixels = env->GetByteArrayElements(pixels_, NULL);
    unsigned char* pixels = (unsigned char*)ipixels;
    jint *it_nums = env->GetIntArrayElements(t_nums_, NULL);
    int* t_nums= (int*)it_nums;
    jint *inums_label = env->GetIntArrayElements(nums_label_, NULL);
    int* nums_label = (int*)inums_label;
    jint *it_devs = env->GetIntArrayElements(t_devs_, NULL);
    int* t_devs = (int*)it_devs;
    jint *idevs_label = env->GetIntArrayElements(devs_label_, NULL);
    int* devs_label = (int*)idevs_label;
    jint *idevs_bipolar = env->GetIntArrayElements(devs_bipolar_, NULL);
    int* devs_bipolar = (int*)idevs_bipolar;


    FILE *file_handle;
    char *kernel_file_buffer, *file_log;
    size_t kernel_file_size, log_size;
    const char* cl_file_name = "/data/local/tmp/ImageProc.cl";
    unsigned char* kernel_name_t = (unsigned  char*)"AdaptiveThreshold";
    unsigned char* kernel_name_c = (unsigned  char*)"Convolution";
    unsigned char* kernel_name_d = (unsigned  char*)"matdotGPU";

    cl_mem d_dst, d_src;
    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel_t;                 // kernel
    cl_kernel kernel_c;                 // kernel
    cl_kernel kernel_d;                 // kernel

    file_handle=fopen(cl_file_name, "r");
    if(file_handle==NULL)
    {
        printf("Couldn't find the file");
        exit(1);
    }

    //read kernel file
    fseek(file_handle, 0, SEEK_END);
    kernel_file_size =ftell(file_handle);
    rewind(file_handle);
    kernel_file_buffer = (char*)malloc(sizeof(char)*(kernel_file_size+1));
    kernel_file_buffer[kernel_file_size]='\0';
    fread(kernel_file_buffer, sizeof(char), kernel_file_size, file_handle);
    fclose(file_handle);

    int i, j, k;

    opencl_infra_creation(context, cpPlatform, device_id, queue, program,
                          kernel_t,kernel_c,kernel_d, kernel_file_buffer, kernel_file_size,
                          kernel_name_t,kernel_name_c,kernel_name_d,
                          d_src, d_dst
    );


    //number recognition
    int data_size = 28;
    unsigned char* sample;
    unsigned char* data = (unsigned char*)malloc(data_size*data_size);
    //unsigned char* p2 = (unsigned char*)malloc(data_size*data_size*4);
    cnn_init(CNN_LETTER); //cnn: letter mode
    for(k=0;k<nums_n;k++){
        int sam_w = (t_nums[k*4+2]+2)-(t_nums[k*4]-2), sam_h = (t_nums[k*4+3]+2)-(t_nums[k*4+1]-2); //addtion margin
        int l;
        int predicted;
        sample = (unsigned char*)malloc(sam_w * sam_h);
        l = 0;
        //for(i=t_nums[k*4+3]-1;i>=t_nums[k*4+1];i--){ //image cutting, reverse
        for(i=t_nums[k*4+1]-2;i<t_nums[k*4+3]+2;i++){ //image cutting
            for(j=t_nums[k*4]-2;j<t_nums[k*4+2]+2;j++){
                sample[l++] = pixels[(i*width+j)*4];
            }
        }
        near_resize(sample, data, sam_w, sam_h, data_size, data_size);
        predicted = predict(context, queue, kernel_c, kernel_d, data);
        free(sample);
        nums_label[k] = predicted;

        for(i=0;i<28;i++){
            char rr[30];
            for(j=0;j<28;j++){
                rr[j]=((data[i*28+j]>0)?'O':'.');
            }
            rr[j] = '\0';
            LOGD("%s",rr);
        }
        LOGD("Predict: %d",predicted);
        //getchar();
    }
    cnn_finite();
    //free(p2);
    free(data);


    //picto recognition
    data_size = 36;
    data = (unsigned char*)malloc(data_size*data_size);
    //p2 = (unsigned char*)malloc(data_size*data_size*4);
    cnn_init(CNN_PICT); //cnn: pict mode
    for(k=0;k<devs_n;k++){
        int sam_w = t_devs[k*4+2]-t_devs[k*4], sam_h = t_devs[k*4+3]-t_devs[k*4+1];
        int l;
        int predicted;
        sample = (unsigned char*)malloc(sam_w * sam_h);
        l = 0;
        //for(i=t_devs[k*4+3]-1;i>=t_devs[k*4+1];i--){ //image cutting, reverse
        for(i=t_devs[k*4+1];i<t_devs[k*4+3];i++){ //image cutting, reverse
            for(j=t_devs[k*4];j<t_devs[k*4+2];j++){
                sample[l++] = pixels[(i*width+j)*4];
            }
        }
        detmid(sample, sam_w, sam_h, &devs_bipolar[k*4]);
        devs_bipolar[k*4] += t_devs[k*4];
        devs_bipolar[k*4+1] += t_devs[k*4+1];
        devs_bipolar[k*4+2] = t_devs[k*4]+sam_w/2;
        devs_bipolar[k*4+3] = t_devs[k*4+1]+sam_h/2;
        near_resize(sample, data, sam_w, sam_h, data_size, data_size);
        /*for(i=0;i<36;i++){
            for(j=0;j<36;j++){
                printf("%c ",(data[i*36+j]>0)?'O':'.');
            }
            printf("\n");
        }*/
        predicted = predict(context, queue, kernel_c, kernel_d, data);
        free(sample);
        devs_label[k] = predicted;

        /*for(i=0;i<data_size*data_size;i++){ p2[i*4] = p2[i*4+1] = p2[i*4+2] = data[i]; p2[i*4+3]=255; }
        write_bmp("data/tmp1.bmp", data_size, data_size, (char *)p2);
        printf("%s\n", (predicted==0)?"Resistor":(predicted==1)?"Input":(predicted==2)?"OUTPUT":"NOLABEL");
        getchar();*/
    }
    cnn_finite();
    //free(p2);
    free(data);

    checkCL(clReleaseProgram(program));
    checkCL(clReleaseKernel(kernel_t));
    checkCL(clReleaseKernel(kernel_c));
    checkCL(clReleaseKernel(kernel_d));
    checkCL(clReleaseCommandQueue(queue));
    checkCL(clReleaseContext(context));

    env->ReleaseByteArrayElements(pixels_, ipixels, 0);
    env->ReleaseIntArrayElements(t_nums_, it_nums, 0);
    env->ReleaseIntArrayElements(nums_label_, inums_label, 0);
    env->ReleaseIntArrayElements(t_devs_, it_devs, 0);
    env->ReleaseIntArrayElements(devs_label_, idevs_label, 0);
    env->ReleaseIntArrayElements(devs_bipolar_, idevs_bipolar, 0);
}


//IO Interface

int fd_switch = 0;
extern "C"
JNIEXPORT jint JNICALL
Java_com_example_uos_project_JNIProcess_openDriverSw(JNIEnv *env, jclass type, jstring path_) {
    const char *path = env->GetStringUTFChars(path_, 0);
    fd_switch = open(path, O_RDONLY);
    env->ReleaseStringUTFChars(path_, path);
    if(fd_switch<0) return -1;
    else return 1;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_uos_project_JNIProcess_closeDriverSw(JNIEnv *env, jclass type) {
    if(fd_switch>0) close(fd_switch);
}

extern "C"
JNIEXPORT jchar JNICALL
Java_com_example_uos_project_JNIProcess_readDriver(JNIEnv *env, jclass type) {
    char ch = 0;
    if(fd_switch>0){
        read(fd_switch,&ch,1);
    }
    return ch;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_uos_project_JNIProcess_getInterrupt(JNIEnv *env, jclass type) {
    int ret = 0;
    char value[100];
    char* ch1 = "Up";
    char* ch2 = "Down";
    char* ch3 = "Left";
    char* ch4 = "Right";
    char* ch5 = "Center";

    ret = read(fd_switch,value,100);
    if(ret<0)
        return -1;
    else{
        if(strcmp(ch1,value) == 0) return 1;
        else if(strcmp(ch2,value) == 0) return 2;
        else if(strcmp(ch3,value) == 0) return 3;
        else if(strcmp(ch4,value) == 0) return 4;
        else if(strcmp(ch5,value) == 0) return 5;
    }
    return 0;
}


int fd_segment = 0;

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_uos_project_JNIProcess_openDriverSe(JNIEnv *env, jclass type,
                                                     jstring path_) {
    const char *path = env->GetStringUTFChars(path_, 0);
    fd_segment=open(path,O_RDWR|O_SYNC);
    env->ReleaseStringUTFChars(path_, path);

    LOGD("%s, %d", path, fd_segment);
    if(fd_segment<0) return -1;
    else return 1;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_uos_project_JNIProcess_closeDriverSe(JNIEnv *env, jclass type) {
    if(fd_segment>0) close(fd_segment);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_uos_project_JNIProcess_writeDriver(JNIEnv *env, jclass type,
                                                    jbyteArray data_, jint length) {
    jbyte *data = env->GetByteArrayElements(data_, NULL);
    if(fd_segment>0) write(fd_segment, (unsigned  char *) data, length);
    env->ReleaseByteArrayElements(data_, data, 0);
}
