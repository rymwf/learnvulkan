/**
 * @file common.cpp
 * @author yangzs
 * @brief 
 * @version 0.1
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include "common.h"

std::string readFile(const char *filename)
{
    std::fstream file(filename, std::ios::in | std::ios::ate | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("failed to open file");
    size_t size0 = file.tellg();
    std::string ret;
    ret.resize(size0);
    file.seekg(0);
    file.read(ret.data(), size0);
    file.close();
    return ret;
}

bool checkValidationLayerSupport()
{
    uint32_t availablelayercount;
    vkEnumerateInstanceLayerProperties(&availablelayercount, nullptr);
    LOG(availablelayercount);
    std::vector<VkLayerProperties> availableLayers(availablelayercount);
    vkEnumerateInstanceLayerProperties(&availablelayercount, &availableLayers[0]);
    for (const char *layername : validationLayers)
    {
        bool flag = true;
        for (auto &p : availableLayers)
        {
            if (strcmp(layername, p.layerName) == 0)
            {
                flag = false;
                break;
            }
        }
        if (flag)
            return false;
    }
    return true;
}

VkInstance createInstance()
{
    if constexpr (enableValidationLayer)
        if (!checkValidationLayerSupport())
            throw std::runtime_error("validation layers requested, but not available");

    VkApplicationInfo appInfo{
        VK_STRUCTURE_TYPE_APPLICATION_INFO,
        NULL,
        "application name",
        VK_MAKE_VERSION(1, 0, 0), //integer, app version
        "engine name",
        VK_MAKE_VERSION(0, 0, 0), //engine version
        VK_API_VERSION_1_1};
    uint32_t glfwExtensionCount;
    auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    LOG(glfwExtensionCount);
    for (uint32_t i = 0; i < glfwExtensionCount; ++i)
        LOG(glfwExtensions[i]);

    LOG("current vulkan version supported extentions:");
    uint32_t extensionCount;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    LOG(extensionCount);
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, &extensions[0]);

    for (auto &e : extensions)
        LOG(e.extensionName);

    VkInstanceCreateInfo instanceInfo{
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, //must
        NULL,
        0, //must
        &appInfo,
        0,
        nullptr,
        glfwExtensionCount,
        glfwExtensions};

    if (enableValidationLayer)
    {
        instanceInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        instanceInfo.ppEnabledLayerNames = validationLayers.data();
    }
    VkInstance instance;
    if (vkCreateInstance(&instanceInfo, 0, &instance) != VK_SUCCESS)
        throw std::runtime_error("create instance failed");
    return instance;
}

int rateDeviceSuitability(VkPhysicalDevice device)
{
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);
    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

    LOG(deviceProperties.apiVersion);
    LOG(VK_VERSION_MAJOR(deviceProperties.apiVersion));
    LOG(VK_VERSION_MINOR(deviceProperties.apiVersion));
    LOG(VK_VERSION_PATCH(deviceProperties.apiVersion));
    LOG(deviceProperties.driverVersion);
    LOG(deviceProperties.vendorID);
    LOG(deviceProperties.deviceID);
    LOG(deviceProperties.deviceType);
    LOG(deviceProperties.deviceName);
    LOG(deviceProperties.pipelineCacheUUID);
    //        LOG(deviceProperties.limits);
    //       LOG(deviceProperties.sparseProperties);

    LOG(deviceFeatures.geometryShader);
    // Application can't function without geometry shaders
    int score = 0;
    if (!deviceFeatures.geometryShader)
    {
        LOG(score);
        LOG(deviceProperties.deviceName);
        return 0;
    }

    //discrete gpu is prefered
    if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        score += 1000;
    // Maximum possible size of textures affects graphics quality
    score += deviceProperties.limits.maxImageDimension2D;

    LOG(score);
    LOG(deviceProperties.deviceName);
    return score;
}
bool isDeviceSuitable(VkPhysicalDevice device)
{
    return true;
}
VkPhysicalDevice pickPhysicalDevice(VkInstance instance)
{
    uint32_t physicalDeviceCount;
    vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr);
    LOG(physicalDeviceCount);
    if (physicalDeviceCount == 0)
    {
        throw std::runtime_error("failed to find GPUs with vulkan support");
    }
    //first element is score
    std::vector<VkPhysicalDevice> devices(physicalDeviceCount);
    vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, devices.data());

    std::vector<std::pair<int, VkPhysicalDevice>> scoreddevices;
    for (auto &device : devices)
    {
        scoreddevices.emplace_back(rateDeviceSuitability(device), device);
    }
    std::sort(scoreddevices.begin(), scoreddevices.end());

    auto &temp = scoreddevices.back();
    if (temp.first == 0)
        throw std::runtime_error("failed to find a suitable GPU");

    return temp.second;
}
std::vector<VkQueueFamilyProperties> queryQueueFamilyProperties(VkPhysicalDevice physicalDevice)
{
    uint32_t queueFamilyPropertyCount;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyPropertyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertyCount, queueFamilyProperties.data());
    LOG(queueFamilyPropertyCount);
    for (uint32_t i = 0; i < queueFamilyPropertyCount; ++i)
    {
        LOG(i);
        LOG(queueFamilyProperties[i].queueFlags);
        LOG(queueFamilyProperties[i].queueCount);
        LOG(queueFamilyProperties[i].timestampValidBits);
        LOG(queueFamilyProperties[i].minImageTransferGranularity.width);
        LOG(queueFamilyProperties[i].minImageTransferGranularity.height);
        LOG(queueFamilyProperties[i].minImageTransferGranularity.depth);
    }
    return queueFamilyProperties;
}

uint32_t findQueueFamilyIndexByFlag(std::vector<VkQueueFamilyProperties> &queueFamilyProperties, VkQueueFlagBits flag, const std::unordered_set<uint32_t> &skipIndices)
{
    std::optional<uint32_t> ret{};
    for (uint32_t i = 0, l = static_cast<uint32_t>(queueFamilyProperties.size()); i < l; ++i)
    {
        if (skipIndices.find(i) != skipIndices.end())
            continue;
        if (queueFamilyProperties[i].queueFlags & flag)
        {
            ret = i;
            break;
        }
    }
    if (!ret.has_value())
        throw std::runtime_error("required queue family not find");
    return ret.value();
}
uint32_t findQueueFamilyIndexPresent(VkPhysicalDevice physicalDevice, uint32_t familyNum, VkSurfaceKHR surface)
{
    std::optional<uint32_t> ret{};
    VkBool32 surfaceSupported = false;
    for (uint32_t i = 0; i < familyNum && !surfaceSupported; ++i)
    {
        vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &surfaceSupported);
        if (surfaceSupported)
            ret = i;
    }
    if (!ret.has_value())
        throw std::runtime_error("required present queue family not find");
    return ret.value();
}

VkDevice createLogicalDevice(VkPhysicalDevice physicalDevice, const std::vector<uint32_t> &queueFamilyIndices)
{
    //show physical device extensions
    uint32_t extensionPropertyCount;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionPropertyCount, nullptr);
    std::vector<VkExtensionProperties> properties(extensionPropertyCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionPropertyCount, properties.data());

    std::vector<const char *> extensionNames;
    extensionNames.reserve(extensionPropertyCount);
    LOG(extensionPropertyCount);
    for (auto &extensionProperty : properties)
    {
        LOG(extensionProperty.extensionName);
        LOG(extensionProperty.specVersion);
        extensionNames.emplace_back(extensionProperty.extensionName);
    }

    //physical device  features
    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);

    const float queuePriorities = 1.0;
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    for (auto i : queueFamilyIndices)
    {
        queueInfos.emplace_back(
            VkDeviceQueueCreateInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                    NULL,
                                    0,
                                    i, //queue family index
                                    1, //queue count
                                    &queuePriorities});
    }

    VkDeviceCreateInfo deviceInfo{
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        NULL,
        0,
        static_cast<uint32_t>(queueInfos.size()),
        queueInfos.data(),
        0, //deprecated
        0, //deprecated
        extensionPropertyCount,
        extensionNames.data(),
        &deviceFeatures};
    VkDevice ret;
    if (vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &ret) != VK_SUCCESS)
        throw std::runtime_error("create logical device failed");

    return ret;
}

std::vector<VkPhysicalDeviceGroupProperties> queryPhysicalDeviceGroupInfo(VkInstance instance)
{
    uint32_t physicalDeviceGroupCount;
    vkEnumeratePhysicalDeviceGroups(instance, &physicalDeviceGroupCount, nullptr);
    std::vector<VkPhysicalDeviceGroupProperties> physicalDeviceGroupProperties(physicalDeviceGroupCount);
    vkEnumeratePhysicalDeviceGroups(instance, &physicalDeviceGroupCount, physicalDeviceGroupProperties.data());
    LOG(physicalDeviceGroupCount);
    for (auto physicalDeviceGroupProperty : physicalDeviceGroupProperties)
    {
        LOG(physicalDeviceGroupProperty.physicalDeviceCount);
    }
    return physicalDeviceGroupProperties;
}

VkSurfaceKHR createWSISurface(VkInstance instance, GLFWwindow *window)
{
    VkSurfaceKHR ret;
#ifdef _WIN32
    if (glfwCreateWindowSurface(instance, window, nullptr, &ret) != VK_SUCCESS)
        throw std::runtime_error("failed to create window surface");
#endif
    return ret;
}

VkSurfaceCapabilitiesKHR querySurfaceCapabilities(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface)
{
    VkSurfaceCapabilitiesKHR surfaceCaps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &surfaceCaps);

    LOG(surfaceCaps.minImageCount);
    LOG(surfaceCaps.maxImageCount);
    LOG(surfaceCaps.currentExtent.width);
    LOG(surfaceCaps.currentExtent.height);
    LOG(surfaceCaps.minImageExtent.width);
    LOG(surfaceCaps.minImageExtent.height);
    LOG(surfaceCaps.maxImageExtent.width);
    LOG(surfaceCaps.maxImageExtent.height);
    LOG(surfaceCaps.maxImageArrayLayers);
    LOG(surfaceCaps.supportedTransforms);
    LOG(surfaceCaps.currentTransform);
    LOG(surfaceCaps.supportedCompositeAlpha);
    LOG(surfaceCaps.supportedUsageFlags);
    //    LOG(surfaceCaps.supportedSurfaceCounters);
    return surfaceCaps;
}
std::vector<VkSurfaceFormatKHR> querySurfaceFormats(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface)
{
    uint32_t surfaceFormatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &surfaceFormatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> surfaceFormats(surfaceFormatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &surfaceFormatCount, surfaceFormats.data());
    LOG(surfaceFormatCount);
    for (auto &surfaceFormat : surfaceFormats)
    {
        LOG(surfaceFormat.colorSpace);
        LOG(surfaceFormat.format);
    }
    return surfaceFormats;
}
std::vector<VkPresentModeKHR> querySurfacePresentModes(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface)
{
    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);
    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, presentModes.data());
    LOG(presentModeCount);
    for (auto &presentMode : presentModes)
        LOG(presentMode);
    return presentModes;
}

VkSwapchainKHR createSwapchain(VkDevice device,
                               VkPhysicalDevice physicalDevice,
                               VkSurfaceKHR surface,
                               const VkSurfaceCapabilitiesKHR &surfaceCaps,
                               const std::vector<uint32_t> &queueFamilyIndices,
                               const VkExtent2D &swapchainExtent,
                               VkFormat &swapchainImageFormatOut)
{
    //set format
    auto surfaceFormat = querySurfaceFormats(physicalDevice, surface);
    VkSurfaceFormatKHR swapchainFormat = surfaceFormat[0]; //default rgba8unorm
    //change to srgba8
    for (auto &e : surfaceFormat)
    {
        if (e.format == VK_FORMAT_R8G8B8A8_SRGB && e.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            swapchainFormat = e;
            break;
        }
    }
    swapchainImageFormatOut = swapchainFormat.format;
    //set presentmode
    auto presentModes = querySurfacePresentModes(physicalDevice, surface);
    VkPresentModeKHR swapchainPresentMode = VK_PRESENT_MODE_FIFO_KHR;
    for (auto &e : presentModes)
    {
        if (e == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            swapchainPresentMode = e;
            break;
        }
    }

    VkSwapchainCreateInfoKHR swapchainInfo{
        VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        NULL,
        0,
        surface,
        surfaceCaps.maxImageCount,
        swapchainFormat.format,
        swapchainFormat.colorSpace,
        swapchainExtent,
        1,                                                                                       //nonstereoscopic-3D application
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,                                                     //if has post process, use VK_IMAGE_USAGE_TRANSFER_DST_BIT
        queueFamilyIndices.size() == 1 ? VK_SHARING_MODE_EXCLUSIVE : VK_SHARING_MODE_CONCURRENT, //a single queue family can access, if to support multiple families, use VK_SHARING_MODE_CONCURRENT
        static_cast<uint32_t>(queueFamilyIndices.size()),
        queueFamilyIndices.data(),
        surfaceCaps.currentTransform,
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        swapchainPresentMode,
        VK_TRUE,
        VK_NULL_HANDLE};

    VkSwapchainKHR ret{};
    if (vkCreateSwapchainKHR(device, &swapchainInfo, nullptr, &ret) != VK_SUCCESS)
        throw std::runtime_error("failed to create swapchain");
    return ret;
}

void createImageViews(VkDevice logicalDevice,
                      VkSwapchainKHR swapchain,
                      std::vector<VkImage> &swapchainImagesOut,
                      std::vector<VkImageView> &swapchainImageViewsOut,
                      VkFormat swapchainImageFormat)
{
    uint32_t imagecount;
    vkGetSwapchainImagesKHR(logicalDevice, swapchain, &imagecount, nullptr);
    LOG(imagecount);
    swapchainImagesOut.resize(imagecount);
    swapchainImageViewsOut.resize(imagecount);
    vkGetSwapchainImagesKHR(logicalDevice, swapchain, &imagecount, swapchainImagesOut.data());

    VkImageViewCreateInfo imageViewCreateInfo{
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        nullptr,
        0,
        0,
        VK_IMAGE_VIEW_TYPE_2D,
        swapchainImageFormat,
        VkComponentMapping{},
        VkImageSubresourceRange{
            VK_IMAGE_ASPECT_COLOR_BIT,
            0,
            1,
            0,
            1}};
    for (uint32_t i = 0; i < imagecount; ++i)
    {
        imageViewCreateInfo.image = swapchainImagesOut[i];
        if (vkCreateImageView(logicalDevice, &imageViewCreateInfo, nullptr, &swapchainImageViewsOut[i]) != VK_SUCCESS)
            throw std::runtime_error("failed to create imageview");
    }
}

VkShaderModule createShaderModule(VkDevice logicalDevice, const std::string &code)
{
    VkShaderModule ret{};
    VkShaderModuleCreateInfo createInfo{
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        nullptr,
        0,
        code.size(),
        reinterpret_cast<const uint32_t *>(code.data())};

    if (vkCreateShaderModule(logicalDevice, &createInfo, nullptr, &ret) != VK_SUCCESS)
        throw std::runtime_error("failed to create shadermodule");
    return ret;
}

VkDescriptorSetLayout createDescriptorSetLayout(VkDevice logicalDevice, const std::vector<VkDescriptorSetLayoutBinding> &setLayoutBindings)
{
    VkDescriptorSetLayout ret;
    VkDescriptorSetLayoutCreateInfo createInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        nullptr,
        0,
        static_cast<uint32_t>(setLayoutBindings.size()),
        setLayoutBindings.data()};

    if (vkCreateDescriptorSetLayout(logicalDevice, &createInfo, nullptr, &ret) != VK_SUCCESS)
        throw std::runtime_error("failded to create decriptor set layout");
    return ret;
}

VkFramebuffer createFramebuffer(VkDevice logicalDevice, VkRenderPass renderpass, const std::vector<VkImageView> &attachments, const VkExtent3D &extent)
{
    VkFramebuffer ret;
    VkFramebufferCreateInfo createInfo{
        VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        nullptr,
        0,
        renderpass,
        static_cast<uint32_t>(attachments.size()),
        attachments.data(),
        extent.width,
        extent.height,
        extent.depth};

    if (vkCreateFramebuffer(logicalDevice, &createInfo, nullptr, &ret) != VK_SUCCESS)
        throw std::runtime_error("failed to create framebuffer");
    return ret;
}
VkPipelineLayout createPipelineLayout(VkDevice logicalDevice, const std::vector<VkDescriptorSetLayout> &setLayouts, const std::vector<VkPushConstantRange> &pushConstantRanges)
{
    VkPipelineLayout ret;
    VkPipelineLayoutCreateInfo createInfo{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        nullptr,
        0,
        static_cast<uint32_t>(setLayouts.size()),
        setLayouts.data(),
        static_cast<uint32_t>(pushConstantRanges.size()),
        pushConstantRanges.data()};

    if (vkCreatePipelineLayout(logicalDevice, &createInfo, nullptr, &ret) != VK_SUCCESS)
        throw std::runtime_error("failed to create pipelinelayout");
    return ret;
}
VkCommandPool createCommandPool(VkDevice logicalDevice, uint32_t queueFamilyIndex)
{
    VkCommandPool ret;

    VkCommandPoolCreateInfo createInfo{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        nullptr,
        0,
        queueFamilyIndex};
    if (vkCreateCommandPool(logicalDevice, &createInfo, nullptr, &ret) != VK_SUCCESS)
        throw std::runtime_error("failed to create command pool");
    return ret;
}
VkSemaphore createSemaphore(VkDevice logicalDevice)
{
    VkSemaphore ret;
    VkSemaphoreCreateInfo createInfo{
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };
    if (vkCreateSemaphore(logicalDevice, &createInfo, nullptr, &ret) != VK_SUCCESS)
        throw std::runtime_error("failed create semaphore");
    return ret;
}

std::vector<VkDisplayPropertiesKHR> queryDisplayProperties(VkPhysicalDevice physicalDevice)
{
    std::vector<VkDisplayPropertiesKHR> ret;
    uint32_t count;
    vkGetPhysicalDeviceDisplayPropertiesKHR(physicalDevice, &count, nullptr);
    ret.resize(count);
    vkGetPhysicalDeviceDisplayPropertiesKHR(physicalDevice, &count, ret.data());
    for (auto &displayProperty : ret)
    {
        LOG(displayProperty.display);
        LOG(displayProperty.displayName);
        LOG(displayProperty.persistentContent);
        LOG(displayProperty.physicalDimensions.width);
        LOG(displayProperty.physicalDimensions.height);
        LOG(displayProperty.physicalResolution.width);
        LOG(displayProperty.physicalResolution.height);
        LOG(displayProperty.planeReorderPossible);
        LOG(displayProperty.supportedTransforms);
    }
    return ret;
}

std::vector<VkDisplayPlanePropertiesKHR> queryDisplayPlaneProperties(VkPhysicalDevice physicalDevice)
{
    std::vector<VkDisplayPlanePropertiesKHR> ret;
    uint32_t count;
    vkGetPhysicalDeviceDisplayPlanePropertiesKHR(physicalDevice, &count, nullptr);
    ret.resize(count);
    vkGetPhysicalDeviceDisplayPlanePropertiesKHR(physicalDevice, &count, ret.data());
    for (auto displayPlaneProperty : ret)
    {
        LOG(displayPlaneProperty.currentDisplay);
        LOG(displayPlaneProperty.currentStackIndex);
    }
    return ret;
}
std::vector<VkDisplayModePropertiesKHR> queryDisplayModeProperties(VkPhysicalDevice physicalDevice, VkDisplayKHR display)
{
    std::vector<VkDisplayModePropertiesKHR> ret;
    uint32_t count;
    vkGetDisplayModePropertiesKHR(physicalDevice, display, &count, nullptr);
    ret.resize(count);
    vkGetDisplayModePropertiesKHR(physicalDevice, display, &count, ret.data());
    for (auto displayModeProperty : ret)
    {
        LOG(displayModeProperty.displayMode);
        LOG(displayModeProperty.parameters.refreshRate);
        LOG(displayModeProperty.parameters.visibleRegion.width);
        LOG(displayModeProperty.parameters.visibleRegion.height);
    }
    return ret;
}
VkFence createFence(VkDevice logicalDevice, VkFenceCreateFlags flags)
{
    VkFence ret;
    VkFenceCreateInfo createInfo{
        VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        nullptr,
        flags};
    if (vkCreateFence(logicalDevice, &createInfo, nullptr, &ret) != VK_SUCCESS)
        throw std::runtime_error("failed to create fence");
    return ret;
}

void chooseSwapExtent(const VkSurfaceCapabilitiesKHR &surfaceCaps, GLFWwindow *window, VkExtent2D &outExtent)
{
    if (surfaceCaps.currentExtent.width != UINT64_MAX)
    {
        outExtent = surfaceCaps.currentExtent;
    }
    else
    {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        VkExtent2D ret{static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
        outExtent.width = (std::max)((std::min)(ret.width, surfaceCaps.maxImageExtent.width),
                                     surfaceCaps.minImageExtent.width);

        outExtent.height = (std::max)((std::min)(ret.height, surfaceCaps.maxImageExtent.height),
                                      surfaceCaps.minImageExtent.height);
    }
}

void createBuffer(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer &outBuffer, VkDeviceMemory &outBufferMemory)
{
    VkBufferCreateInfo createInfo{
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        nullptr,
        0,
        size,
        usage,
        VK_SHARING_MODE_EXCLUSIVE,
    };
    if (vkCreateBuffer(logicalDevice, &createInfo, nullptr, &outBuffer) != VK_SUCCESS)
        throw std::runtime_error("failed to create vertex buffer");

    VkMemoryRequirements vertexBufferMemoryRequirements;
    vkGetBufferMemoryRequirements(logicalDevice, outBuffer, &vertexBufferMemoryRequirements);

    LOG(vertexBufferMemoryRequirements.size);
    LOG(vertexBufferMemoryRequirements.alignment);
    LOG(vertexBufferMemoryRequirements.memoryTypeBits); //typebits is the memorytype indices in physical memory properties

    auto memoryTypeIndex = findMemoryTypeIndex(physicalDevice, vertexBufferMemoryRequirements.memoryTypeBits, properties); //the index of memory type

    outBufferMemory = allocateMemory(logicalDevice, memoryTypeIndex, vertexBufferMemoryRequirements.size);

    vkBindBufferMemory(logicalDevice, outBuffer, outBufferMemory, 0);
}

uint32_t findMemoryTypeIndex(VkPhysicalDevice physicalDevice, uint32_t typeIndexFilter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
#if 0
    LOG(memoryProperties.memoryHeapCount);
    for (uint32_t i = 0; i < memoryProperties.memoryHeapCount; i++)
    {
        LOG(i);
        LOG(memoryProperties.memoryHeaps[i].flags);
        LOG(memoryProperties.memoryHeaps[i].size);
    }
    LOG(memoryProperties.memoryTypeCount);
#endif
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
    {
#if 0
        LOG(i);
        LOG(memoryProperties.memoryTypes[i].heapIndex);
        LOG(memoryProperties.memoryTypes[i].propertyFlags);
#endif
        if (typeIndexFilter & (1 << i) && properties & memoryProperties.memoryTypes[i].propertyFlags)
            return i;
    }
    throw std::runtime_error("failed to find suitable memory type");
}

VkDeviceMemory allocateMemory(VkDevice logicalDevice, uint32_t memoryTypeIndex, VkDeviceSize memsize)
{
    VkMemoryAllocateInfo allocateInfo{
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        nullptr,
        memsize,
        memoryTypeIndex};

    VkDeviceMemory allocatedMemory;
    if (vkAllocateMemory(logicalDevice, &allocateInfo, nullptr, &allocatedMemory) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate vertex memory");
    return allocatedMemory;
}

std::vector<VkCommandBuffer> createCommandBuffers(VkDevice logicalDevice, VkCommandPool commandPool, uint32_t count)
{
    std::vector<VkCommandBuffer> commandBuffers;
    commandBuffers.resize(count);
    VkCommandBufferAllocateInfo allocateInfo{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        nullptr,
        commandPool,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        count};
    if (vkAllocateCommandBuffers(logicalDevice, &allocateInfo, commandBuffers.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to create command buffer");

    return commandBuffers;
}

void copyBuffer(VkDevice logicalDevice, VkCommandPool commandPool, VkQueue queue, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize bufferSize)
{
    auto commandBuffers = createCommandBuffers(logicalDevice, commandPool, 1);
    VkCommandBufferBeginInfo cmdBeginInfo{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VkBufferCopy bufferCopyRegion{0, 0, bufferSize};
    vkBeginCommandBuffer(commandBuffers[0], &cmdBeginInfo);
    vkCmdCopyBuffer(commandBuffers[0], srcBuffer, dstBuffer, 1, &bufferCopyRegion);
    vkEndCommandBuffer(commandBuffers[0]);

    VkSubmitInfo submitInfo{
        VK_STRUCTURE_TYPE_SUBMIT_INFO,
    };
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = commandBuffers.data();
    vkQueueSubmit(queue, 1, &submitInfo, 0);
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(logicalDevice, commandPool, 1, commandBuffers.data());
}
