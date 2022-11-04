#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <string>
#include <vector>

#include "shader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <CL/cl.h>
#include <CL/opencl.h>
#include <CL/cl_ext.h>
#include <CL/cl_gl.h>
#include <windows.h>

using namespace std;

typedef unsigned int uint;

void processTimeStep(void);
void renderFrame(void);
bool initFluidState(const char* imagePath);
bool stepFluidState();
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
void lbmProgram();

static void error_callback(int error, const char* desc)
{
    fputs(desc, stderr);
}

static void key_callback(GLFWwindow* wind, int key, int scancode, int action, int mods)
{
    // TODO:
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    //! make sure the viewport matches the new window dimensions; note that width and 
    //! height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

typedef struct {
    cl_device_id* d;
    cl_command_queue q;
    cl_program p;
    cl_kernel k;
    GLuint tex;
    size_t dims[3];
} process_params;


//! global variables
float tau = 0.58;
int winWidth = 0, winHeight = 0;
size_t VECTOR_SIZE;
//! those data will be used in shaders
unsigned int lbmBuffer[3];
//!	lbmBoundary stores boundary
unsigned int lbmBoundary;
unsigned int textureFrame;
//! cl parameters
process_params params;
//! host ptrs
float* lbmData, * boundaryData;
//! opencl frame
cl_mem cl_frame;
float* e_global;

//OpenCL kernel which is run for every work item created.
const char* saxpy_kernel =
"__kernel                                   \n"
"void saxpy_kernel(float alpha,     \n"
"                  __global float *A,       \n"
"                  __global float *B,       \n"
"                  __global float *C)       \n"
"{                                          \n"
"    //Get the index of the work-item       \n"
"    int index = get_global_id(0);          \n"
"    C[index] = alpha* A[index] + B[index]; \n"
"}                                          \n";
int main()
{
    //! glfw: initialize and configure
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    //! glfw window creation
    GLFWwindow* window = glfwCreateWindow(800, 600, "CSCI5390 Tut", NULL, NULL);
    if (window == NULL)
    {
        cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    //! set vsync, make sure the simulation won't go too fast
    glfwSwapInterval(1);
    //! glad: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    //------------------------------------------------------------------------
    //                              CL                                       =
    //------------------------------------------------------------------------
    // Get platform info
    cl_int clStatus;
    cl_uint num_platforms;
    clStatus = clGetPlatformIDs(0, 0, &num_platforms);
    if (clStatus != CL_SUCCESS) {
        std::cerr << "Unable to get platforms\n";
        return 0;
    }
    std::vector<cl_platform_id> platforms(num_platforms);
    clStatus = clGetPlatformIDs(num_platforms, &platforms[0], &num_platforms);
    if (clStatus != CL_SUCCESS) {
        std::cerr << "Unable to get platform ID\n";
        return 0;
    }

    // Get the devices list and choose the device you want to run on
    cl_uint num_devices;
    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    params.d = (cl_device_id*)
        malloc(sizeof(cl_device_id) * num_devices);
    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, params.d, NULL);


    // Create context
    cl_context_properties context_properties[] =
    {
        CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
        CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
        CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[0],
        NULL
    };
    cl_context context = clCreateContext(context_properties, 1, &params.d[0], NULL, NULL, &clStatus);
    if (context == 0) {
        printf("Can’t create OpenCL context %d\n", clStatus);
        return 0;
    }

    // Create a command queue and use the first device
    params.q = clCreateCommandQueueWithProperties(context, params.d[0], 0, &clStatus);


    //------------------------------------------------------------------------
    //                              GL                                       =
    //------------------------------------------------------------------------
    // 
    // 
    //! ---------build and compile our shader program---------
    //! vertex shader and fragment shader
    Shader renderProgram("./vertex.vert", "./render.frag");

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float vertices[] = {
        //! positions       //! texture coordinates
        1.0f,  1.0f, 0.0f,  1.0f, 1.0f, //! top right
        1.0f, -1.0f, 0.0f,  1.0f, 0.0f, //! bottom right
       -1.0f, -1.0f, 0.0f,  0.0f, 0.0f, //! bottom left
       -1.0f,  1.0f, 0.0f,  0.0f, 1.0f  //! top left 
    };
    unsigned int indices[] = {
        0, 1, 3, //! first triangle
        1, 2, 3  //! second triangle
    };
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    //! vertex attribute: position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    //! vertex attribute: texture coordinates
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    //! load and create textures
    // -------------------------
    const char* image_path = "./mask.jpg";
    if (!initFluidState(image_path))
    {
        cout << "Error: state initialization failed!" << endl;
        return -1;
    }

    float* textureData = new float[winWidth * winHeight * 4]; // rgbv
    for (int y = 0; y < winHeight; y++)
    {
        for (int x = 0; x < winWidth; x++)
        {
            int index = y * winWidth + x;
            textureData[4 * index + 0] = 1;
            textureData[4 * index + 1] = 1;
            textureData[4 * index + 2] = 1;
            textureData[4 * index + 3] = 0;
        }
    }
    glGenTextures(1, &textureFrame);
    glBindTexture(GL_TEXTURE_2D, textureFrame);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, winWidth, winHeight, 0, GL_RGBA, GL_FLOAT, textureData);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureFrame, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    //------------------------------------------------------------------------
    //                              CL BUFFER                                =
    //------------------------------------------------------------------------
    // Create memory buffers on the device for each vector
    cl_mem cl_boundaryData = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, 3 * VECTOR_SIZE * sizeof(float), boundaryData, &clStatus);
    cl_mem cl_lbmData = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, 3 * 4 * VECTOR_SIZE * sizeof(float), lbmData, &clStatus);
    /* cl_mem cl_state_texture2 = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, 3 * VECTOR_SIZE * sizeof(float), state_texture2, &clStatus);
     cl_mem cl_state_texture3 = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, 3 * VECTOR_SIZE * sizeof(float), state_texture3, &clStatus);*/


    // //------------------------------------------------------------------------
    // //                              DEBUG                                    =
    // //------------------------------------------------------------------------
    // // Create memory buffers on the device for each vector
    //VECTOR_SIZE = 64;
    //float* A = (float*)malloc(sizeof(float) * VECTOR_SIZE);
    //float* B = (float*)malloc(sizeof(float) * VECTOR_SIZE);
    //float* C = (float*)malloc(sizeof(float) * VECTOR_SIZE);
    //for (int i = 0; i < VECTOR_SIZE; i++)
    //{
    //    A[i] = i;
    //    B[i] = VECTOR_SIZE - i;
    //    C[i] = 0;
    //}
    //A[0] = 8;
    //cl_mem A_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, VECTOR_SIZE * sizeof(float), NULL, &clStatus);
    //cl_mem B_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, VECTOR_SIZE * sizeof(float), NULL, &clStatus);
    //cl_mem C_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, VECTOR_SIZE * sizeof(float), NULL, &clStatus);
    //cl_frame = clCreateFromGLTexture(context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, textureFrame, &clStatus);


    //// Copy the Buffer A and B to the device
    //clStatus = clEnqueueWriteBuffer(params.q, A_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), A, 0, NULL, NULL);
    //clStatus = clEnqueueWriteBuffer(params.q, B_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), B, 0, NULL, NULL);

    //cl_program program = clCreateProgramWithSource(context, 1, (const char**)&saxpy_kernel, NULL, &clStatus);

    //clStatus = clBuildProgram(program, 1, params.d, NULL, NULL, NULL);
    //params.k = clCreateKernel(program, "saxpy_kernel", &clStatus);
    ////------------------------------------------------------------------------
    ////                              DEBUG                                    =
    ////------------------------------------------------------------------------

    // Create a program from the kernel source
    // First step is to load the kernel in a local memory
    FILE* f = fopen("./lbm.cl", "rb");
    fseek(f, 0, SEEK_END);
    size_t programSize = ftell(f);
    rewind(f);
    //load kernel into buffer
    char* programBuffer = (char*)malloc(programSize + 1);
    programBuffer[programSize] = '\0';
    fread(programBuffer, sizeof(char), programSize, f);
    fclose(f);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&programBuffer, NULL, &clStatus);
    clStatus = clBuildProgram(program, 1, params.d, NULL, NULL, NULL);
    if (clStatus)
    {
        printf("Error: Failed to read source! %d\n", clStatus);
        return EXIT_FAILURE;
    }

    // Build the program
    clStatus = clBuildProgram(program, 1, params.d, NULL, NULL, NULL);
    if (clStatus)
    {
        printf("Error: Failed to build program! %d\n", clStatus);
        return EXIT_FAILURE;
    }

    // Create the OpenCL kernel
    params.k = clCreateKernel(program, "lbm_kernel", &clStatus);
    if (clStatus)
    {
        printf("Error: Failed to create kernel! %d\n", clStatus);
        return EXIT_FAILURE;
    }

    // copy from gl buffer
    cl_frame = clCreateFromGLTexture(context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, textureFrame, &clStatus);


    //! set uniform variables for render.frag
    renderProgram.use(); // don't forget to activate/use the shader before setting uniforms!
    glUniform1i(glGetUniformLocation(renderProgram.ID, "boundary_texture"), 0);
    glUniform1i(glGetUniformLocation(renderProgram.ID, "state_texture3"), 1);

    //! render loop
    // ------------
    while (!glfwWindowShouldClose(window))
    {
        //! input
        processInput(window);

        //! render to screen
        // -----------------
        //! now bind back to default frame buffer and draw a quad plane with the attached frame_buffer color texture
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClearColor(199.0 / 255, 237.0 / 255, 204.0 / 255, 1.0f); // set clear color to white (not really necessary actually, since we won't be able to see behind the quad anyways)
        glClear(GL_COLOR_BUFFER_BIT);

        clStatus = clEnqueueAcquireGLObjects(params.q, 1, &cl_frame, 0, 0, 0);
        if (clStatus)
        {
            printf("Error: Failed to fetch gl object! %d\n", clStatus);
            return EXIT_FAILURE;
        }
        // Set the arguments of the kernel
        clStatus = clSetKernelArg(params.k, 0, sizeof(cl_mem), (void*)&cl_lbmData);
        if (clStatus)
        {
            printf("Error: Failed to set kernel arguments! %d\n", clStatus);
            return EXIT_FAILURE;
        }
        clStatus |= clSetKernelArg(params.k, 1, sizeof(cl_mem), (void*)&cl_boundaryData);
        clStatus |= clSetKernelArg(params.k, 2, sizeof(cl_mem), (void*)&cl_frame);
        clStatus |= clSetKernelArg(params.k, 3, sizeof(int), &winWidth);
        clStatus |= clSetKernelArg(params.k, 4, sizeof(int), &winHeight);
        clStatus |= clSetKernelArg(params.k, 5, sizeof(float), &tau);
        clStatus |= clSetKernelArg(params.k, 6, sizeof(float), &e_global);
        

        // Execute the OpenCL kernel on the list
        size_t global_size = VECTOR_SIZE; //
        size_t local_size = 64;           //
        clStatus = clEnqueueNDRangeKernel(params.q, params.k, 2, NULL, &global_size, NULL, 0, NULL, NULL);
        if (clStatus)
        {
            printf("Error: Failed to execute kernel! %d\n", clStatus);
            return EXIT_FAILURE;
        }

        // Clean up and wait for all the comands to complete.
        clStatus = clFlush(params.q);
        clStatus = clFinish(params.q);
        clStatus = clEnqueueReleaseGLObjects(params.q, 1, &cl_frame, 0, 0, 0);

        //// Set the arguments of the kernel
        //// Set the arguments of the kernel
        //float alpha = 2.0;
        //clStatus = clEnqueueAcquireGLObjects(params.q, 1, &cl_frame, 0, 0, 0);
        //if (clStatus)
        //{
        //    printf("Error: Failed to fetch gl object! %d\n", clStatus);
        //    return EXIT_FAILURE;
        //}
        //clStatus = clSetKernelArg(params.k, 0, sizeof(float), (void*)&alpha);
        //clStatus |= clSetKernelArg(params.k, 1, sizeof(cl_mem), (void*)&A_clmem);
        //clStatus |= clSetKernelArg(params.k, 2, sizeof(cl_mem), (void*)&B_clmem);
        //clStatus |= clSetKernelArg(params.k, 3, sizeof(cl_mem), (void*)&C_clmem);
        //if (clStatus)
        //{
        //    printf("Error: Failed to set kernel arguments!\n");
        //    return EXIT_FAILURE;
        //}


        //// Execute the OpenCL kernel on the list
        //size_t global_size = VECTOR_SIZE; // Process the entire lists
        //size_t local_size = 64;           // Process one item at a time
        //clStatus = clEnqueueNDRangeKernel(params.q, params.k, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
        //if (clStatus)
        //{
        //    printf("Error: Failed to execute kernel!\n");
        //    return EXIT_FAILURE;
        //}

        //clStatus = clEnqueueReadBuffer(params.q, C_clmem, CL_TRUE, 0, sizeof(float) * VECTOR_SIZE, C, 0, NULL, NULL);

        //// Read the cl memory C_clmem on device to the host variable C
        //// Clean up and wait for all the comands to complete.
        //clStatus = clFlush(params.q);
        //clStatus = clFinish(params.q);

        //for (int i = 0; i < VECTOR_SIZE; i++)
        //    printf("%f * %f + %f = %f\n", alpha, A[i], B[i], C[i]);

        exit(1);
        renderProgram.use();
        glBindVertexArray(VAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, lbmBoundary);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, textureFrame);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        //! glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}


bool initFluidState(const char* imagePath)
{
    //! load image
    int nrChannels;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* maskData = stbi_load(imagePath, &winWidth, &winHeight, &nrChannels, 0);
    cout << "texture image (HxW):" << winHeight << " x " << winWidth << endl;
    boundaryData = new float[winWidth * winHeight * 3]; // 3 channels
    VECTOR_SIZE = winWidth * winHeight;
    if (boundaryData == NULL) {
        cout << "Unable to allocate memory!" << endl;
        return false;
    }
    //!	Fill _boundaryData_ with image data from _boundaryBitmap_
    for (int y = 0; y < winHeight; y++)
    {
        for (int x = 0; x < winWidth; x++)
        {
            int index = y * winWidth + x;
            //! Pixels near image margin are set to be boundary 
            if ((x < 2) || (x > (winWidth - 3)) || (y < 2) || (y > (winHeight - 3))) // boundaries
            {
                boundaryData[3 * index + 0] = 0.0f;
                boundaryData[3 * index + 1] = 0.0f;
                boundaryData[3 * index + 2] = 0.0f;
            }
            else
            {
                //! pixels: 0.0 or 1.0
                unsigned char r = maskData[3 * index + 0];
                unsigned char g = maskData[3 * index + 1];
                unsigned char b = maskData[3 * index + 2];
                boundaryData[3 * index + 0] = r / 255.0;
                boundaryData[3 * index + 1] = g / 255.0;
                boundaryData[3 * index + 2] = b / 255.0;
            }
        }
    }
    stbi_image_free(maskData);
    //!	generate OpenGL texture buffer for Boundary data
    glGenTextures(1, &lbmBoundary);
    glBindTexture(GL_TEXTURE_2D, lbmBoundary);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, winWidth, winHeight, 0, GL_RGB, GL_FLOAT, boundaryData);

    //! initialize the data buffers for LBM simulation use
    lbmData = new float[winWidth * winHeight * 12]; // 12 channels

    //! initial state of the lbmBuffer[] for simulation
    //!	macroscopic velocity: <ux,uy>
    //!	macroscopic density:  rho
    //! initialize distribution function
    float w[9] = { 4.0f / 9.0f,1.0f / 9.0f,1.0f / 9.0f,1.0f / 9.0f,1.0f / 9.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f };
    float e[9][2] = { { 0,0 },{ 1,0 },{ 0,1 },{ -1,0 },{ 0,-1 },{ 1,1 },{ -1,1 },{ -1,-1 },{ 1,-1 } };
    e_global = new float[9];
    for (int i = 0; i < 9; i++) {
        e_global[i * 2 + 0] = e[i][0];
        e_global[i * 2 + 1] = e[i][1];
    }
    //!	initialize values of f0-f8, rho, ux, uy for each pixel
    for (int y = 0; y < winHeight; y++)
    {
        for (int x = 0; x < winWidth; x++)
        {
            float ux = 0.3;
            float uy = 0.06;
            float rho = 1.0;
            float uu_dot = (ux * ux + uy * uy);
            float f[9];
            for (int i = 0; i < 9; i++)
            {
                float eu_dot = (e[i][0] * ux + e[i][1] * uy);
                f[i] = w[i] * rho * (1.0f + 3.0f * eu_dot + 4.5f * eu_dot * eu_dot - 1.5f * uu_dot);
            }
            int index = y * winWidth + x;
            //! f1~f4
            lbmData[12 * index + 0] = f[1];
            lbmData[12 * index + 1] = f[2];
            lbmData[12 * index + 2] = f[3];
            lbmData[12 * index + 3] = f[4];
            //! f5~f8
            lbmData[12 * index + 4] = f[5];
            lbmData[12 * index + 5] = f[6];
            lbmData[12 * index + 6] = f[7];
            lbmData[12 * index + 7] = f[8];
            //! f0, rho, and (ux,uy)
            lbmData[12 * index + 8] = f[0];
            lbmData[12 * index + 9] = rho;;
            lbmData[12 * index + 10] = ux;
            lbmData[12 * index + 11] = uy;
        }
    }

    return true;
}
