/**
 * @file 01-triangle.cpp
 * @author yangzs
 * @brief 
 * @version 0.1
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include "common.h"

#define MAX_FRAMES_IN_FLIGHT 2

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

    VkDevice logicalDevice;

    VkQueue graphicQueue;
    VkQueue presentQueue;

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

    //each frame should have its own set of semaphores
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> imagesInFlight;

    size_t currentFrame{0};

    void initWindow()
    {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, __FILE__, nullptr, nullptr);
    }

    void initVulkan()
    {
        instance = createInstance();

        physicalDevice = pickPhysicalDevice(instance);

        queryPhysicalDeviceGroupInfo(instance);

        //auto displayProperties = queryDisplayProperties(physicalDevice);
        //auto displayPlaneProperties = queryDisplayPlaneProperties(physicalDevice);

        surface = createWSISurface(instance, window);
        surfaceCaps = querySurfaceCapabilities(physicalDevice, surface);

        findSuitableQueueFamilyIndices();

        std::unordered_set<uint32_t> tempQueueFamilyIndices{graphicQueueFamilyIndex, presentQueueFamilyIndex};
        std::vector<uint32_t> queueFamilyIndices(tempQueueFamilyIndices.begin(), tempQueueFamilyIndices.end());
        logicalDevice = createLogicalDevice(physicalDevice, queueFamilyIndices);

        vkGetDeviceQueue(logicalDevice, graphicQueueFamilyIndex, 0, &graphicQueue);
        vkGetDeviceQueue(logicalDevice, presentQueueFamilyIndex, 0, &presentQueue);

        chooseSwapExtent(surfaceCaps, window, swapchainExtent);
        swapchain = createSwapchain(logicalDevice, physicalDevice, surface, surfaceCaps, queueFamilyIndices, swapchainExtent, swapchainImageFormat);

        createSwapchainImageViews(logicalDevice, swapchain, swapchainImages, swapchainImageViews, swapchainImageFormat);

        createShaderModuleInfos();

        createDescriptors();

        pipelineLayout = createPipelineLayout(logicalDevice, descriptorSetLayouts);

        createRenderPass();

        createGraphicsPipeline();

        createFramebuffers();

        commandPool = createCommandPool(logicalDevice, graphicQueueFamilyIndex);

        createCommandBuffers();

        createSemaphores();

        createFences();
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            drawFrame();
        }
        vkDeviceWaitIdle(logicalDevice);
    }

    void cleanup()
    {
        for (auto v : inFlightFences)
            vkDestroyFence(logicalDevice, v, nullptr);
        for (auto v : imageAvailableSemaphores)
            vkDestroySemaphore(logicalDevice, v, nullptr);
        for (auto v : renderFinishedSemaphores)
            vkDestroySemaphore(logicalDevice, v, nullptr);

        vkDestroyCommandPool(logicalDevice, commandPool, nullptr);

        for (auto e : swapchainFramebuffers)
            vkDestroyFramebuffer(logicalDevice, e, nullptr);

        for (auto e : descriptorSetLayouts)
            vkDestroyDescriptorSetLayout(logicalDevice, e, nullptr);

        vkDestroyPipeline(logicalDevice, graphicsPipeline, nullptr);

        vkDestroyRenderPass(logicalDevice, renderPass, nullptr);

        vkDestroyPipelineLayout(logicalDevice, pipelineLayout, nullptr);

        for (auto &e : shaderModuleInfos)
            vkDestroyShaderModule(logicalDevice, e.shaderModule, nullptr);

        for (auto i : swapchainImageViews)
            vkDestroyImageView(logicalDevice, i, nullptr);

        vkDestroySwapchainKHR(logicalDevice, swapchain, nullptr);

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

        LOG(graphicQueueFamilyIndex);
        LOG(presentQueueFamilyIndex);
    }

    void createShaderModuleInfos()
    {
        //vertex shader
        //build by run compileshaders.py
        auto vertShaderFile = WORKING_DIR "01.vert.spv";
        auto fragShaderFile = WORKING_DIR "01.frag.spv";

        auto vertShaderCode = readFile(vertShaderFile);
        auto fragShaderCode = readFile(fragShaderFile);

        auto vertShaderModule = createShaderModule(logicalDevice, vertShaderCode);
        auto fragShaderModule = createShaderModule(logicalDevice, fragShaderCode);

        shaderModuleInfos.emplace_back(ShaderModuleInfo{VK_SHADER_STAGE_VERTEX_BIT, vertShaderModule});
        shaderModuleInfos.emplace_back(ShaderModuleInfo{VK_SHADER_STAGE_FRAGMENT_BIT, fragShaderModule});
    };

    void createDescriptors()
    {
        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {};
        descriptorSetLayouts.emplace_back(createDescriptorSetLayout(logicalDevice, setLayoutBindings));
    }

    void createRenderPass()
    {
        std::vector<VkAttachmentDescription> attachmentDescription{
            {0,
             swapchainImageFormat,
             VK_SAMPLE_COUNT_1_BIT,
             VK_ATTACHMENT_LOAD_OP_CLEAR,
             VK_ATTACHMENT_STORE_OP_DONT_CARE,
             VK_ATTACHMENT_LOAD_OP_DONT_CARE,
             VK_ATTACHMENT_STORE_OP_DONT_CARE,
             VK_IMAGE_LAYOUT_UNDEFINED,
             VK_IMAGE_LAYOUT_PRESENT_SRC_KHR},
            //            {0,
            //             VK_FORMAT_D24_UNORM_S8_UINT,
            //             VK_SAMPLE_COUNT_1_BIT,
            //             VK_ATTACHMENT_LOAD_OP_CLEAR,
            //             VK_ATTACHMENT_STORE_OP_DONT_CARE,
            //             VK_ATTACHMENT_LOAD_OP_CLEAR,
            //             VK_ATTACHMENT_STORE_OP_DONT_CARE,
            //             VK_IMAGE_LAYOUT_UNDEFINED,
            //             VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL},
        };

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
                nullptr,
                //&depthStencilAttachment,
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

        VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo{
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            nullptr,
            0,
            0,
            nullptr,
            0,
            nullptr};
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
            0};

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
        swapchainFramebuffers.reserve(swapchainImageViews.size());
        for (auto &swapchainImageView : swapchainImageViews)
        {
            std::vector<VkImageView> attachments{swapchainImageView};
            swapchainFramebuffers.emplace_back(createFramebuffer(logicalDevice, renderPass, attachments, VkExtent3D{swapchainExtent.width, swapchainExtent.height, 1}));
        }
    }
    void createCommandBuffers()
    {
        uint32_t count = static_cast<uint32_t>(swapchainFramebuffers.size());

        commandBuffers.resize(count);
        VkCommandBufferAllocateInfo allocateInfo{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            nullptr,
            commandPool,
            VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            count};
        if (vkAllocateCommandBuffers(logicalDevice, &allocateInfo, commandBuffers.data()) != VK_SUCCESS)
            throw std::runtime_error("failed to create command buffer");

        VkClearValue clearValue{
            {
                0.2f,
                0.2f,
                0.2f,
                1,
            }};
        VkCommandBufferBeginInfo cmdBeginInfo{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VkRenderPassBeginInfo renderPassBeginInfo{
            VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            nullptr,
            renderPass,
            0,
            {0, 0, swapchainExtent.width, swapchainExtent.height},
            1,
            &clearValue};

        for (uint32_t i = 0; i < count; ++i)
        {
            if (vkBeginCommandBuffer(commandBuffers[i], &cmdBeginInfo) != VK_SUCCESS)
                throw std::runtime_error("failed to begin recording command buffer");
            renderPassBeginInfo.framebuffer = swapchainFramebuffers[i];

            vkCmdBeginRenderPass(commandBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

            vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

            vkCmdDraw(commandBuffers[i], 3, 1, 0, 0);

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
        vkAcquireNextImageKHR(logicalDevice, swapchain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

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
        vkQueuePresentKHR(presentQueue, &presentInfo);

        //prevent reusing semaphore, but the whole graphic pipeline is only used for one frame at a time
        //vkQueueWaitIdle(presentQueue);

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
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
