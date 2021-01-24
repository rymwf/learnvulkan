/**
 * @file 03-image.cpp
 * @author yangzs
 * @brief 
 * @version 0.1
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include "common.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

constexpr int MAX_FRAMES_IN_FLIGHT = 2;
constexpr char *testImagePath = PROJECT_DIR "/assets/textures/Lenna_test.jpg";

constexpr char *vertShaderFile = WORKING_DIR "/04.vert.spv";
constexpr char *fragShaderFile = WORKING_DIR "/04.frag.spv";

//std140, round to base alignment of vec4
struct UBO_MVP
{
    alignas(16) glm::mat4 M;
    alignas(16) glm::mat4 V;
    alignas(16) glm::mat4 P;
};

struct Vertex
{
    glm::vec2 pos;
    glm::vec3 color;
    glm::vec2 texCoord;
    static VkVertexInputBindingDescription getBindingDescription()
    {
        return VkVertexInputBindingDescription{
            0,
            sizeof Vertex,
            VK_VERTEX_INPUT_RATE_VERTEX};
    }
    static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions()
    {
        return std::vector<VkVertexInputAttributeDescription>{
            VkVertexInputAttributeDescription{0,
                                              0,
                                              VK_FORMAT_R32G32_SFLOAT,
                                              static_cast<uint32_t>(offsetof(Vertex, pos))},
            VkVertexInputAttributeDescription{1,
                                              0,
                                              VK_FORMAT_R32G32B32_SFLOAT,
                                              static_cast<uint32_t>(offsetof(Vertex, color))},
            VkVertexInputAttributeDescription{2,
                                              0,
                                              VK_FORMAT_R32G32_SFLOAT,
                                              static_cast<uint32_t>(offsetof(Vertex, texCoord))},
        };
    }
};
std::vector<Vertex> vertices{
    {{-1, -1}, {1, 0, 0}, {0, 0}},
    {{1, -1}, {0, 1, 0}, {1, 0}},
    {{1, 1}, {0, 0, 1}, {1, 1}},
    {{-1, 1}, {1, 1, 1}, {0, 1}},
};

std::vector<uint16_t> indices{
    0, 1, 2, 2, 3, 0};

struct InstanceAttribute
{
    glm::mat4 translations;
    static VkVertexInputBindingDescription getBindingDescription()
    {
        return VkVertexInputBindingDescription{
            1,
            sizeof(InstanceAttribute),
            VK_VERTEX_INPUT_RATE_INSTANCE};
    }
    static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions()
    {
        return std::vector<VkVertexInputAttributeDescription>{
            VkVertexInputAttributeDescription{
                3,
                1,
                VK_FORMAT_R32G32B32A32_SFLOAT,
                0},
            VkVertexInputAttributeDescription{
                4,
                1,
                VK_FORMAT_R32G32B32A32_SFLOAT,
                16},
            VkVertexInputAttributeDescription{
                5,
                1,
                VK_FORMAT_R32G32B32A32_SFLOAT,
                32},
            VkVertexInputAttributeDescription{
                6,
                1,
                VK_FORMAT_R32G32B32A32_SFLOAT,
                48},
        };
    }
};
std::vector<InstanceAttribute> instanceAttributes{
    {glm::translate(glm::rotate(glm::mat4(1), glm::radians(15.f), glm::vec3(0, 1, 0)), glm::vec3(-0.5))},
    {glm::translate(glm::rotate(glm::mat4(1), glm::radians(15.f), glm::vec3(0, 1, 0)), glm::vec3(0.5))},
};

class HelloTriangleApplication
{
public:
    void run()
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow *window;
    VkInstance instance;

    VkPhysicalDevice physicalDevice;

    uint32_t graphicQueueFamilyIndex;
    uint32_t presentQueueFamilyIndex;
    uint32_t transferQueueFamilyIndex;

    std::vector<uint32_t> queueFamilyIndices; //contains only unique indices, include graphicQueueFamilyIndex and presentQueueFamilyIndex,

    VkDevice logicalDevice;

    VkQueue graphicQueue;
    VkQueue presentQueue;
    VkQueue transferQueue;

    VkSurfaceKHR surface;
    VkSurfaceCapabilitiesKHR surfaceCaps;

    VkSwapchainKHR swapchain;
    VkExtent2D swapchainExtent;
    VkFormat swapchainImageFormat;

    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;

    std::vector<ShaderModuleInfo> shaderModuleInfos;

    std::vector<VkDescriptorSetLayout> descriptorSetLayouts;

    VkPipelineLayout pipelineLayout;
    VkRenderPass renderPass;

    VkPipeline graphicsPipeline;

    std::vector<VkFramebuffer> swapchainFramebuffers;

    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
    VkCommandPool transferCommandPool;

    //each frame should have its own set of semaphores
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> imagesInFlight;

    bool framebufferResized{false};

    size_t currentFrame{0};

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    std::vector<VkBuffer> uboMVPBuffers;
    std::vector<VkDeviceMemory> uboMVPBufferMemorys;
    VkBuffer instanceBuffer;
    VkDeviceMemory instanceBufferMemory;

    VkImage testImage;
    VkDeviceMemory testImageMemory;
    VkImageView testImageView;

    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;

    VkSampler sampler;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    uint32_t FPS;
    float frameTimeInterval_ms;
    std::chrono::system_clock::time_point curTime;
    std::chrono::system_clock::time_point startTime;

    UBO_MVP uboMVP;

    void initWindow()
    {
        startTime = std::chrono::system_clock::now();
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(WIDTH, HEIGHT, __FILE__, nullptr, nullptr);

        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow *window, int width, int height)
    {
        auto app = static_cast<HelloTriangleApplication *>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void initVulkan()
    {
        instance = createInstance();
        physicalDevice = pickPhysicalDevice(instance);
        queryPhysicalDeviceGroupInfo(instance);

        //auto displayProperties = queryDisplayProperties(physicalDevice);
        //auto displayPlaneProperties = queryDisplayPlaneProperties(physicalDevice);

        surface = createWSISurface(instance, window);

        findSuitableQueueFamilyIndices();

        std::unordered_set<uint32_t> tempQueueFamilyIndices{graphicQueueFamilyIndex, presentQueueFamilyIndex, transferQueueFamilyIndex};
        queueFamilyIndices.insert(queueFamilyIndices.end(), tempQueueFamilyIndices.begin(), tempQueueFamilyIndices.end());
        logicalDevice = createLogicalDevice(physicalDevice, queueFamilyIndices);
        vkGetDeviceQueue(logicalDevice, graphicQueueFamilyIndex, 0, &graphicQueue);
        vkGetDeviceQueue(logicalDevice, presentQueueFamilyIndex, 0, &presentQueue);
        vkGetDeviceQueue(logicalDevice, transferQueueFamilyIndex, 0, &transferQueue);

        commandPool = createCommandPool(logicalDevice, graphicQueueFamilyIndex);
        transferCommandPool = createCommandPool(logicalDevice, transferQueueFamilyIndex);

        createShaderModuleInfos();
        createVertexBuffer();
        createInstanceAttribeBuffer();
        createIndexBuffer();
        createTextureImage();
        createImageTextureView();

        surfaceCaps = querySurfaceCapabilities(physicalDevice, surface);
        chooseSwapExtent(surfaceCaps, window, swapchainExtent);
        swapchain = createSwapchain(logicalDevice, physicalDevice, surface, surfaceCaps, queueFamilyIndices, swapchainExtent, swapchainImageFormat);
        createSwapchainImageViews(logicalDevice, swapchain, swapchainImages, swapchainImageViews, swapchainImageFormat);

        createDepthResources();

        createUBObuffer();
        createDescriptors();
        pipelineLayout = createPipelineLayout(logicalDevice, descriptorSetLayouts);

        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        submitCommandBuffers();

        createSemaphores();
        createFences();
    }
    void recreateSwapchain()
    {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        if (width == 0 || height == 0)
        {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(logicalDevice);
        cleanupSwapchain();
        surfaceCaps = querySurfaceCapabilities(physicalDevice, surface);
        chooseSwapExtent(surfaceCaps, window, swapchainExtent);
        swapchain = createSwapchain(logicalDevice, physicalDevice, surface, surfaceCaps, queueFamilyIndices, swapchainExtent, swapchainImageFormat);
        createSwapchainImageViews(logicalDevice, swapchain, swapchainImages, swapchainImageViews, swapchainImageFormat);
        createDepthResources();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        submitCommandBuffers();
        framebufferResized = false;
    }
    void cleanupSwapchain()
    {

        vkDestroyImage(logicalDevice, depthImage, nullptr);
        vkFreeMemory(logicalDevice, depthImageMemory, nullptr);
        vkDestroyImageView(logicalDevice, depthImageView, nullptr);

        for (auto e : swapchainFramebuffers)
            vkDestroyFramebuffer(logicalDevice, e, nullptr);

        //free current command buffers to reuse current command pool
        vkFreeCommandBuffers(logicalDevice, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());

        vkDestroyPipeline(logicalDevice, graphicsPipeline, nullptr);

        vkDestroyRenderPass(logicalDevice, renderPass, nullptr);

        for (auto i : swapchainImageViews)
            vkDestroyImageView(logicalDevice, i, nullptr);

        vkDestroySwapchainKHR(logicalDevice, swapchain, nullptr);
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window))
        {
            auto temptime = std::chrono::system_clock::now();
            frameTimeInterval_ms = std::chrono::duration<float, std::chrono::milliseconds::period>(temptime - curTime).count();
            curTime = temptime;
            FPS = static_cast<uint32_t>(1000 / frameTimeInterval_ms);
            glfwPollEvents();
            drawFrame();
        }
        vkDeviceWaitIdle(logicalDevice);
    }

    void cleanup()
    {
        cleanupSwapchain();

        vkDestroyBuffer(logicalDevice, instanceBuffer, nullptr);
        vkFreeMemory(logicalDevice, instanceBufferMemory, nullptr);
        vkDestroyBuffer(logicalDevice, vertexBuffer, nullptr);
        vkFreeMemory(logicalDevice, vertexBufferMemory, nullptr);
        vkDestroyBuffer(logicalDevice, indexBuffer, nullptr);
        vkFreeMemory(logicalDevice, indexBufferMemory, nullptr);

        vkDestroyImage(logicalDevice, testImage, nullptr);
        vkFreeMemory(logicalDevice, testImageMemory, nullptr);
        vkDestroyImageView(logicalDevice, testImageView, nullptr);
        vkDestroySampler(logicalDevice, sampler, nullptr);

        for (size_t i = 0, len = uboMVPBuffers.size(); i < len; ++i)
        {
            vkDestroyBuffer(logicalDevice, uboMVPBuffers[i], nullptr);
            vkFreeMemory(logicalDevice, uboMVPBufferMemorys[i], nullptr);
        }

        vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);

        for (auto v : inFlightFences)
            vkDestroyFence(logicalDevice, v, nullptr);
        for (auto v : imageAvailableSemaphores)
            vkDestroySemaphore(logicalDevice, v, nullptr);
        for (auto v : renderFinishedSemaphores)
            vkDestroySemaphore(logicalDevice, v, nullptr);

        vkDestroyCommandPool(logicalDevice, commandPool, nullptr);
        vkDestroyCommandPool(logicalDevice, transferCommandPool, nullptr);

        std::unordered_set<VkDescriptorSetLayout> tempsetlayout(descriptorSetLayouts.begin(), descriptorSetLayouts.end());
        for (auto e : tempsetlayout)
            vkDestroyDescriptorSetLayout(logicalDevice, e, nullptr);

        vkDestroyPipelineLayout(logicalDevice, pipelineLayout, nullptr);

        for (auto &e : shaderModuleInfos)
            vkDestroyShaderModule(logicalDevice, e.shaderModule, nullptr);

        vkDestroySurfaceKHR(instance, surface, nullptr);

        vkDestroyDevice(logicalDevice, nullptr);

        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);
        glfwTerminate();
    }
    void findSuitableQueueFamilyIndices()
    {
        auto queueFamilyProperties = queryQueueFamilyProperties(physicalDevice);

        graphicQueueFamilyIndex = findQueueFamilyIndexByFlag(queueFamilyProperties, VK_QUEUE_GRAPHICS_BIT);
        presentQueueFamilyIndex = findQueueFamilyIndexPresent(physicalDevice, static_cast<uint32_t>(queueFamilyProperties.size()), surface);
        transferQueueFamilyIndex = findQueueFamilyIndexByFlag(queueFamilyProperties, VK_QUEUE_TRANSFER_BIT, {graphicQueueFamilyIndex, presentQueueFamilyIndex});

        LOG(graphicQueueFamilyIndex);
        LOG(presentQueueFamilyIndex);
        LOG(transferQueueFamilyIndex);
    }

    void createShaderModuleInfos()
    {
        //vertex shader
        //build by run compileshaders.py

        auto vertShaderCode = readFile(vertShaderFile);
        auto fragShaderCode = readFile(fragShaderFile);

        auto vertShaderModule = createShaderModule(logicalDevice, vertShaderCode);
        auto fragShaderModule = createShaderModule(logicalDevice, fragShaderCode);

        shaderModuleInfos.emplace_back(ShaderModuleInfo{VK_SHADER_STAGE_VERTEX_BIT, vertShaderModule});
        shaderModuleInfos.emplace_back(ShaderModuleInfo{VK_SHADER_STAGE_FRAGMENT_BIT, fragShaderModule});
    };

    void createDescriptors()
    {
        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
            {
                0,
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                1,
                VK_SHADER_STAGE_VERTEX_BIT,
                nullptr //image sampling related descriptors
            },
            {1,
             VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
             1,
             VK_SHADER_STAGE_FRAGMENT_BIT,
             nullptr}};
        uint32_t count = static_cast<uint32_t>(swapchainImages.size());
        //descriptorSetLayouts.insert(descriptorSetLayouts.end(), count, createDescriptorSetLayout(logicalDevice, setLayoutBindings));
        descriptorSetLayouts.insert(descriptorSetLayouts.end(), count, createDescriptorSetLayout(logicalDevice, setLayoutBindings));

        std::vector<VkDescriptorPoolSize> poolSizes{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                                     count},
                                                    {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                     count}};
        descriptorPool = createDescriptorPool(logicalDevice, count, poolSizes);

        descriptorSets = createDescriptorSets(logicalDevice, descriptorPool, descriptorSetLayouts);

        for (uint32_t i = 0; i < count; i++)
        {
            VkDescriptorBufferInfo bufferInfo{
                uboMVPBuffers[i],
                0,
                sizeof(UBO_MVP)};
            VkDescriptorImageInfo imageInfo{
                sampler,
                testImageView,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            std::vector<VkWriteDescriptorSet> writeSets{{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                         nullptr,
                                                         descriptorSets[i],
                                                         0,
                                                         0,
                                                         1,
                                                         VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                                         nullptr,
                                                         &bufferInfo,
                                                         nullptr},
                                                        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                         nullptr,
                                                         descriptorSets[i],
                                                         1,
                                                         0,
                                                         1,
                                                         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                         &imageInfo,
                                                         nullptr,
                                                         nullptr}};
            vkUpdateDescriptorSets(logicalDevice, static_cast<uint32_t>(writeSets.size()), writeSets.data(), 0, nullptr);
        }
    }

    void createRenderPass()
    {
        std::vector<VkAttachmentDescription> attachmentDescription{
            {0,
             swapchainImageFormat,
             VK_SAMPLE_COUNT_1_BIT,
             VK_ATTACHMENT_LOAD_OP_CLEAR,
             VK_ATTACHMENT_STORE_OP_DONT_CARE,
             VK_ATTACHMENT_LOAD_OP_CLEAR,
             VK_ATTACHMENT_STORE_OP_DONT_CARE,
             VK_IMAGE_LAYOUT_UNDEFINED,
             VK_IMAGE_LAYOUT_PRESENT_SRC_KHR},
            {0,
             findDepthFormat(),
             VK_SAMPLE_COUNT_1_BIT,
             VK_ATTACHMENT_LOAD_OP_CLEAR,
             VK_ATTACHMENT_STORE_OP_DONT_CARE,
             VK_ATTACHMENT_LOAD_OP_DONT_CARE,
             VK_ATTACHMENT_STORE_OP_DONT_CARE,
             VK_IMAGE_LAYOUT_UNDEFINED,
             VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL}};

        std::vector<VkAttachmentReference> colorAttachments{
            {0, //the index of attachment description
             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}};
        VkAttachmentReference depthStencilAttachment{
            1,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

        std::vector<VkSubpassDescription> subPasses{
            {
                0,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                0,
                nullptr, //input attachments, read from a shader
                static_cast<uint32_t>(colorAttachments.size()),
                colorAttachments.data(),
                nullptr, //resolve attachment, used for multisampling color attachment
                &depthStencilAttachment,
                0,
                nullptr //preserve attachments, when data must be preserved
            }};

        std::vector<VkSubpassDependency> subPassDependencies{
            {
                VK_SUBPASS_EXTERNAL,
                0,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                0,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            }};

        VkRenderPassCreateInfo createInfo{
            VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            nullptr,
            0,
            static_cast<uint32_t>(attachmentDescription.size()),
            attachmentDescription.data(),
            static_cast<uint32_t>(subPasses.size()),
            subPasses.data(),
            static_cast<uint32_t>(subPassDependencies.size()),
            subPassDependencies.data()};

        if (vkCreateRenderPass(logicalDevice, &createInfo, nullptr, &renderPass) != VK_SUCCESS)
            throw std::runtime_error("failed to create renderpass");
    }

    void createGraphicsPipeline()
    {
        std::vector<VkPipelineShaderStageCreateInfo> shaderStageCreateInfos;
        shaderStageCreateInfos.reserve(shaderModuleInfos.size());

        VkPipelineShaderStageCreateInfo shaderStageCreateInfo{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            VK_SHADER_STAGE_VERTEX_BIT,
            0,
            "main"};

        for (auto &shaderModuleInfo : shaderModuleInfos)
        {
            shaderStageCreateInfo.stage = shaderModuleInfo.shaderStageFlagBits;
            shaderStageCreateInfo.module = shaderModuleInfo.shaderModule;
            shaderStageCreateInfos.emplace_back(shaderStageCreateInfo);
        }

        std::vector<VkVertexInputBindingDescription> inputBindingDescriptions{
            Vertex::getBindingDescription(),
            InstanceAttribute::getBindingDescription()};
        auto inputAttributeDescriptions = Vertex::getAttributeDescriptions();
        auto instanceAttribesDesc = InstanceAttribute::getAttributeDescriptions();
        inputAttributeDescriptions.insert(inputAttributeDescriptions.end(), instanceAttribesDesc.begin(), instanceAttribesDesc.end());

        VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo{
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            nullptr,
            0,
            static_cast<uint32_t>(inputBindingDescriptions.size()),
            inputBindingDescriptions.data(),
            static_cast<uint32_t>(inputAttributeDescriptions.size()),
            inputAttributeDescriptions.data()};

        VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCreateInfo{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            nullptr,
            0,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            VK_FALSE //primitive restart
        };
        VkPipelineTessellationStateCreateInfo tessellationStateCreateInfo{
            VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
            nullptr,
            0};

        std::vector<VkViewport> viewports{
            {
                0, 0,                                                                                  //x,y upperleft corner
                static_cast<float>(swapchainExtent.width), static_cast<float>(swapchainExtent.height), //viewport size,
                0,                                                                                     //mindepth
                1                                                                                      //maxdepth
            }};
        std::vector<VkRect2D> scissors = {
            {{0, 0}, //offset
             swapchainExtent}};
        VkPipelineViewportStateCreateInfo viewportStateCreateInfo{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            nullptr,
            0,
            static_cast<uint32_t>(viewports.size()),
            viewports.data(),
            static_cast<uint32_t>(scissors.size()),
            scissors.data()};
        VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo{
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            nullptr,
            0,
            VK_FALSE,
            VK_FALSE,
            VK_POLYGON_MODE_FILL,
            //VK_CULL_MODE_NONE,
            VK_CULL_MODE_BACK_BIT,
            VK_FRONT_FACE_COUNTER_CLOCKWISE,
            VK_FALSE, //depth bias
            0,        //depth bias constant factor
            0,        //depth bias clamp
            0,        //depth bias slope factor
            1         //line width
        };
        VkPipelineMultisampleStateCreateInfo multisampleStateCreateInfo{
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            nullptr,
            0,
            VK_SAMPLE_COUNT_1_BIT,
            VK_FALSE, //disable sample shading
            1,        //min sample shading, must be in range [0,1], can be ignored when sample shading is diabled ,
            nullptr,  //sample mask
            VK_FALSE,
            VK_FALSE, //
        };
        VkPipelineDepthStencilStateCreateInfo depthStencilStateCreateInfo{
            VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            nullptr,
            0,
            VK_TRUE,
            VK_TRUE,
            VK_COMPARE_OP_LESS,
            VK_FALSE,
            VK_FALSE,
        };

        std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachmentStates{
            {VK_TRUE,
             VK_BLEND_FACTOR_SRC_ALPHA,
             VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
             VK_BLEND_OP_ADD,
             VK_BLEND_FACTOR_SRC_ALPHA,
             VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
             VK_BLEND_OP_ADD,
             VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT}};
        VkPipelineColorBlendStateCreateInfo colorBlendStateCreateInfo{
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            nullptr,
            0,
            VK_TRUE,
            VK_LOGIC_OP_COPY,
            static_cast<uint32_t>(colorBlendAttachmentStates.size()),
            colorBlendAttachmentStates.data(),
            {0, 0, 0, 0} //blend constant
        };
        std::vector<VkDynamicState> dynamicStates{
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_LINE_WIDTH};
        VkPipelineDynamicStateCreateInfo dynamicStateCreateInfo{
            VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            nullptr,
            0,
            static_cast<uint32_t>(dynamicStates.size()),
            dynamicStates.data()};

        VkGraphicsPipelineCreateInfo pipelineCreateInfo{
            VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            nullptr,
            0,
            static_cast<uint32_t>(shaderStageCreateInfos.size()),
            shaderStageCreateInfos.data(),
            &vertexInputStateCreateInfo,
            &inputAssemblyStateCreateInfo,
            &tessellationStateCreateInfo,
            &viewportStateCreateInfo,
            &rasterizationStateCreateInfo,
            &multisampleStateCreateInfo,
            &depthStencilStateCreateInfo,
            &colorBlendStateCreateInfo,
            nullptr, //&dynamicStateCreateInfo,
            pipelineLayout,
            renderPass,
            0, //subpass index in renderpass

        };

        std::vector<VkGraphicsPipelineCreateInfo> graphicsPipelineCreateInfos{pipelineCreateInfo};

        if (vkCreateGraphicsPipelines(logicalDevice,
                                      VK_NULL_HANDLE,
                                      static_cast<uint32_t>(graphicsPipelineCreateInfos.size()),
                                      graphicsPipelineCreateInfos.data(),
                                      nullptr,
                                      &graphicsPipeline) != VK_SUCCESS)
            throw std::runtime_error("failed to create graphics pipeline");
    }
    void createFramebuffers()
    {
        swapchainFramebuffers.resize(swapchainImageViews.size());
        for (size_t i = 0, l = swapchainImages.size(); i < l; ++i)
        {
            std::vector<VkImageView> attachments{swapchainImageViews[i], depthImageView};
            swapchainFramebuffers[i] = createFramebuffer(logicalDevice, renderPass, attachments, VkExtent3D{swapchainExtent.width, swapchainExtent.height, 1});
        }
    }
    void submitCommandBuffers()
    {
        auto count = static_cast<uint32_t>(swapchainFramebuffers.size());
        commandBuffers = createCommandBuffers(logicalDevice, commandPool, count);

        std::array<VkClearValue, 2> clearValues;
        clearValues[0].color = {0.2f, 0.2f, 0.2f, 1};
        clearValues[1].depthStencil = {1.f, 0};

        VkCommandBufferBeginInfo cmdBeginInfo{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VkRenderPassBeginInfo renderPassBeginInfo{
            VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            nullptr,
            renderPass,
            0,
            {0, 0, swapchainExtent.width, swapchainExtent.height},
            static_cast<uint32_t>(clearValues.size()),
            clearValues.data()};

        for (uint32_t i = 0; i < count; ++i)
        {
            if (vkBeginCommandBuffer(commandBuffers[i], &cmdBeginInfo) != VK_SUCCESS)
                throw std::runtime_error("failed to begin recording command buffer");
            renderPassBeginInfo.framebuffer = swapchainFramebuffers[i];

            vkCmdBeginRenderPass(commandBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

            vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
            std::vector<VkBuffer> vertexBuffers = {vertexBuffer, instanceBuffer};
            VkDeviceSize offsets[] = {0, 0};
            vkCmdBindVertexBuffers(commandBuffers[i], 0, static_cast<uint32_t>(vertexBuffers.size()), vertexBuffers.data(), offsets);
            vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT16);

            vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);

            //            vkCmdDraw(commandBuffers[i], static_cast<uint32_t>(vertices.size()), 1, 0, 0);
            vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(indices.size()), static_cast<uint32_t>(instanceAttributes.size()), 0, 0, 0);

            vkCmdEndRenderPass(commandBuffers[i]);

            if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS)
                throw std::runtime_error("failed to end commandbuffer");
        }
    }

    void createSemaphores()
    {
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            imageAvailableSemaphores.emplace_back(createSemaphore(logicalDevice));
            renderFinishedSemaphores.emplace_back(createSemaphore(logicalDevice));
        }
    }

    void createFences()
    {
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            //initialize signaled fence
            inFlightFences.emplace_back(createFence(logicalDevice, VK_FENCE_CREATE_SIGNALED_BIT));
        }
        imagesInFlight.resize(swapchainImages.size());
    }

    void drawFrame()
    {

        vkWaitForFences(logicalDevice, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex;

        //acquire an available presentable image, timeout(nanoseconds)
        auto res = vkAcquireNextImageKHR(logicalDevice, swapchain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
        if (res == VK_ERROR_OUT_OF_DATE_KHR)
        {
            recreateSwapchain();
            return;
        }
        else if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
        {
            throw std::runtime_error("failed to acquire swapchain image");
        }

        updateUBObuffer(imageIndex);

        // Check if a previous frame is using this image (i.e. there is its fence to wait on)
        if (imagesInFlight[imageIndex] != VK_NULL_HANDLE)
        {
            vkWaitForFences(logicalDevice, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
        }
        imagesInFlight[imageIndex] = inFlightFences[currentFrame];

        vkResetFences(logicalDevice, 1, &inFlightFences[currentFrame]);

        std::vector<VkSemaphore> waitSemaphores{imageAvailableSemaphores[currentFrame]};
        std::vector<VkPipelineStageFlags> waitDstStages{VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        std::vector<VkSemaphore> signalSemaphores{renderFinishedSemaphores[currentFrame]};

        std::vector<VkSubmitInfo> submitInfos{
            {VK_STRUCTURE_TYPE_SUBMIT_INFO,
             nullptr,
             static_cast<uint32_t>(waitSemaphores.size()),
             waitSemaphores.data(),
             waitDstStages.data(),
             1,
             &commandBuffers[imageIndex],
             static_cast<uint32_t>(signalSemaphores.size()),
             signalSemaphores.data()}};

        if (vkQueueSubmit(graphicQueue, static_cast<uint32_t>(submitInfos.size()), submitInfos.data(), inFlightFences[currentFrame]) != VK_SUCCESS)
            throw std::runtime_error("failed to submit draw command");

        std::vector<VkSwapchainKHR> swapchains{swapchain};

        VkPresentInfoKHR presentInfo{
            VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            nullptr,
            static_cast<uint32_t>(signalSemaphores.size()),
            signalSemaphores.data(),
            static_cast<uint32_t>(swapchains.size()),
            swapchains.data(),
            &imageIndex,
        };
        res = vkQueuePresentKHR(presentQueue, &presentInfo);
        if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR || framebufferResized)
        {
            recreateSwapchain();
        }
        else if (res != VK_SUCCESS)
        {
            throw std::runtime_error("failed to present swapchain image");
        }

        //prevent reusing semaphore, but the whole graphic pipeline is only used for one frame at a time
        //vkQueueWaitIdle(presentQueue);

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
    void createVertexBuffer()
    {
        VkDeviceSize buffersize = sizeof(vertices[0]) * vertices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(physicalDevice, logicalDevice, buffersize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
        void *data;
        vkMapMemory(logicalDevice, stagingBufferMemory, 0, buffersize, 0, &data);
        memcpy(data, vertices.data(), buffersize);
        vkUnmapMemory(logicalDevice, stagingBufferMemory);

        //use device local buffer is fastest
        createBuffer(physicalDevice, logicalDevice, buffersize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

        auto cmdBuffer = beginOneTimeCommands(logicalDevice, transferCommandPool);
        VkBufferCopy bufferCopyRegion{0, 0, buffersize};
        vkCmdCopyBuffer(cmdBuffer, stagingBuffer, vertexBuffer, 1, &bufferCopyRegion);
        endOneTimeCommands(logicalDevice, transferQueue, transferCommandPool, cmdBuffer);

        vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
    }
    void createInstanceAttribeBuffer()
    {
        VkDeviceSize buffersize = sizeof(instanceAttributes[0]) * instanceAttributes.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(physicalDevice, logicalDevice, buffersize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
        void *data;
        vkMapMemory(logicalDevice, stagingBufferMemory, 0, buffersize, 0, &data);
        memcpy(data, instanceAttributes.data(), buffersize);
        vkUnmapMemory(logicalDevice, stagingBufferMemory);

        //use device local buffer is fastest
        createBuffer(physicalDevice, logicalDevice, buffersize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, instanceBuffer, instanceBufferMemory);

        auto cmdBuffer = beginOneTimeCommands(logicalDevice, transferCommandPool);
        VkBufferCopy bufferCopyRegion{0, 0, buffersize};
        vkCmdCopyBuffer(cmdBuffer, stagingBuffer, instanceBuffer, 1, &bufferCopyRegion);
        endOneTimeCommands(logicalDevice, transferQueue, transferCommandPool, cmdBuffer);

        vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
    }
    void createIndexBuffer()
    {
        VkDeviceSize buffersize = sizeof(indices[0]) * indices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(physicalDevice, logicalDevice, buffersize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
        void *data;
        vkMapMemory(logicalDevice, stagingBufferMemory, 0, buffersize, 0, &data);
        memcpy(data, indices.data(), buffersize);
        vkUnmapMemory(logicalDevice, stagingBufferMemory);

        //use device local buffer is fastest
        createBuffer(physicalDevice, logicalDevice, buffersize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

        auto cmdBuffer = beginOneTimeCommands(logicalDevice, transferCommandPool);
        VkBufferCopy bufferCopyRegion{0, 0, buffersize};
        vkCmdCopyBuffer(cmdBuffer, stagingBuffer, indexBuffer, 1, &bufferCopyRegion);
        endOneTimeCommands(logicalDevice, transferQueue, transferCommandPool, cmdBuffer);

        vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
    }
    void createUBObuffer()
    {
        uboMVP = {
            glm::mat4(1),
            glm::lookAt(glm::vec3{0, 0, -3}, glm::vec3{0}, glm::vec3{0, -1, 0}),
            glm::perspective(glm::radians(60.f), float(swapchainExtent.width) / swapchainExtent.height, 1.f, 10.f)};

        auto count = swapchainImages.size();
        uboMVPBuffers.resize(count);
        uboMVPBufferMemorys.resize(count);
        VkDeviceSize buffersize = sizeof(UBO_MVP);
        for (size_t i = 0; i < count; ++i)
        {
            createBuffer(physicalDevice, logicalDevice, buffersize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, uboMVPBuffers[i], uboMVPBufferMemorys[i]);
            void *data;
            vkMapMemory(logicalDevice, uboMVPBufferMemorys[i], 0, buffersize, 0, &data);
            memcpy(data, &uboMVP, buffersize);
            vkUnmapMemory(logicalDevice, uboMVPBufferMemorys[i]);
        }
    }

    void updateUBObuffer(uint32_t index)
    {
        //    float time = std::chrono::duration<float, std::chrono::seconds::period>(curTime - startTime).count();
        //    uboMVP.M = glm::rotate(glm::mat4(1), time * glm::radians(90.f), glm::vec3(0, 1, 0));
        //    uboMVP.V = glm::lookAt(glm::vec3{0, 0, 5}, glm::vec3{0}, glm::vec3{0, -1, 0}); //{{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
        //    uboMVP.P = glm::perspective(glm::radians(45.f), float(swapchainExtent.width) / swapchainExtent.height, 1.f, 10.f);

        //    uint32_t buffersize = sizeof(UBO_MVP);
        //    void *data;
        //    vkMapMemory(logicalDevice, uboMVPBufferMemorys[index], 0, buffersize, 0, &data);
        //    memcpy(data, &uboMVP, buffersize);
        //    vkUnmapMemory(logicalDevice, uboMVPBufferMemorys[index]);
    }

    void createTextureImage()
    {
        int width, height, components;
        stbi_set_flip_vertically_on_load(true);
        auto pixels = stbi_load(testImagePath, &width, &height, &components, 4); //force load an alpha channel,even not exist
        VkDeviceSize imageSize = width * height * 4;

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(physicalDevice, logicalDevice, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void *data;
        vkMapMemory(logicalDevice, stagingBufferMemory, 0, imageSize, 0, &data);
        memcpy(data, pixels, imageSize);
        vkUnmapMemory(logicalDevice, stagingBufferMemory);

        stbi_image_free(pixels);

        VkImageCreateInfo imageCreateInfo{
            VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            nullptr,
            0,
            VK_IMAGE_TYPE_2D,
            //            VK_FORMAT_R8G8B8A8_SRGB,
            VK_FORMAT_R8G8B8A8_UNORM,
            {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1},
            1,
            1,
            VK_SAMPLE_COUNT_1_BIT,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_SHARING_MODE_EXCLUSIVE,
            0,
            nullptr, //ignore
            VK_IMAGE_LAYOUT_UNDEFINED};

        createImage(physicalDevice, logicalDevice, imageCreateInfo, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, testImage, testImageMemory);

        auto cmdBuffer = beginOneTimeCommands(logicalDevice, commandPool);
        //1. transfer layout from undefined to transder destination

        VkImageMemoryBarrier barrier{
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            0,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            testImage,
            {VK_IMAGE_ASPECT_COLOR_BIT,
             0,
             1,
             0,
             1}};
        vkCmdPipelineBarrier(cmdBuffer,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0,
                             0, nullptr,
                             0, nullptr,
                             1, &barrier);
        //2. copy buffer to image
        VkBufferImageCopy region{
            0,
            0,
            0,
            {VK_IMAGE_ASPECT_COLOR_BIT,
             0,
             0,
             1},
            {},
            {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1}};
        vkCmdCopyBufferToImage(cmdBuffer, stagingBuffer, testImage, barrier.newLayout, 1, &region);

        //3. transfer layout from trander destiation to shader reading
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        vkCmdPipelineBarrier(cmdBuffer,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0,
                             0, nullptr,
                             0, nullptr,
                             1, &barrier);

        endOneTimeCommands(logicalDevice, graphicQueue, commandPool, cmdBuffer);
        vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
    }
    void createImageTextureView()
    {
        testImageView = createImageView(logicalDevice, testImage, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
        sampler = createSampler(logicalDevice, VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR);
    }
    void createDepthResources()
    {
        VkFormat depthFormat = findDepthFormat();
        VkImageCreateInfo createInfo{
            VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            nullptr,
            0,
            VK_IMAGE_TYPE_2D,
            depthFormat,
            {swapchainExtent.width, swapchainExtent.height, 1},
            1,
            1,
            VK_SAMPLE_COUNT_1_BIT,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_SHARING_MODE_EXCLUSIVE,
            0,
            nullptr,
            VK_IMAGE_LAYOUT_UNDEFINED};

        createImage(physicalDevice, logicalDevice, createInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
        depthImageView = createImageView(logicalDevice, depthImage, VK_IMAGE_VIEW_TYPE_2D, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
    }
    VkFormat findDepthFormat()
    {
        return findSupportedFormat(physicalDevice, {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT}, VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }
};

int main()
{
    HelloTriangleApplication app;

    try
    {
        app.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
