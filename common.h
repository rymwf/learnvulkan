/**
 * @file common.h
 * @author yangzs
 * @brief 
 * @version 0.1
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#pragma once
#include <iostream>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <algorithm>
#include <optional>
#include <fstream>
#include <filesystem>
#include <unordered_set>
#include <array>
#include <cstring>
#include <chrono>
#include <unordered_map>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#include "config.h"

#define GLFW_INCLUDE_VULKAN

#ifdef _WIN32
#include <Windows.h>
#include <GLFW/glfw3.h>
#include <vulkan/vulkan_win32.h>
#elif __APPLE__
#include <vulkan/vulkan_macos.h>
#elif __linux__
#elif __ANDROID__
#endif

#ifdef NDEBUG
#define LOG(str)
#else
#define LOG(str) \
    std::cout << __FILE__ << " " << __LINE__ << ":  " << #str << ": " << str << std::endl
#endif

#ifdef NDEBUG
constexpr bool enableValidationLayer = false;
#else
constexpr bool enableValidationLayer = true;
#endif

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;

const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

struct ShaderModuleInfo
{
    VkShaderStageFlagBits shaderStageFlagBits;
    VkShaderModule shaderModule;
};

struct Vertex
{
    glm::vec3 pos;
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
                                              VK_FORMAT_R32G32B32_SFLOAT,
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
bool operator==(const Vertex &lhv, const Vertex &rhv);

namespace std
{
    template <>
    struct hash<Vertex>
    {
        std::size_t operator()(Vertex const &vertex) const noexcept
        {
            return ((hash<glm::vec3>{}(vertex.pos) ^ hash<glm::vec3>{}(vertex.color) << 1) >> 1) ^ (hash<glm::vec2>{}(vertex.texCoord) << 1);
        }
    };
}; // namespace std

//query funtions
VkSurfaceCapabilitiesKHR querySurfaceCapabilities(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);
std::vector<VkSurfaceFormatKHR> querySurfaceFormats(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);
std::vector<VkPresentModeKHR> querySurfacePresentModes(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);
std::vector<VkQueueFamilyProperties> queryQueueFamilyProperties(VkPhysicalDevice physicalDevice);
std::vector<VkDisplayPropertiesKHR> queryDisplayProperties(VkPhysicalDevice physicalDevice);
std::vector<VkDisplayPlanePropertiesKHR> queryDisplayPlaneProperties(VkPhysicalDevice physicalDevice);
std::vector<VkDisplayModePropertiesKHR> queryDisplayModeProperties(VkPhysicalDevice physicalDevice, VkDisplayKHR display);

std::string readFile(const char *filename);

VkInstance createInstance();

VkPhysicalDevice pickPhysicalDevice();

bool isDeviceSuitable(VkPhysicalDevice device);

int rateDeviceSuitability(VkPhysicalDevice device);

VkPhysicalDevice pickPhysicalDevice(VkInstance instance);

std::vector<VkPhysicalDeviceGroupProperties> queryPhysicalDeviceGroupInfo(VkInstance instance);

VkDevice createLogicalDevice(VkPhysicalDevice physicalDevice, const std::vector<uint32_t> &queueFamilyIndices);

VkSurfaceKHR createWSISurface(VkInstance instance, GLFWwindow *window);

VkSwapchainKHR createSwapchain(VkDevice device,
                               VkPhysicalDevice physicalDevice,
                               VkSurfaceKHR surface,
                               const VkSurfaceCapabilitiesKHR &surfaceCaps,
                               const std::vector<uint32_t> &queueFamilyIndices,
                               const VkExtent2D &swapchainExtent,
                               VkFormat &swapchainImageFormatOut);

uint32_t findQueueFamilyIndexByFlag(std::vector<VkQueueFamilyProperties> &queueFamilyProperties, VkQueueFlagBits flag, const std::unordered_set<uint32_t> &skipIndices = {});
uint32_t findQueueFamilyIndexPresent(VkPhysicalDevice physicalDevice, uint32_t familyNum, VkSurfaceKHR surface);

void createSwapchainImageViews(VkDevice logicalDevice,
                               VkSwapchainKHR swapchain,
                               std::vector<VkImage> &swapchainImagesOut,
                               std::vector<VkImageView> &swapchainImageViewsOut,
                               VkFormat swapchainImageFormat);

VkShaderModule createShaderModule(VkDevice logicalDevice, const std::string &code);

VkDescriptorSetLayout createDescriptorSetLayout(VkDevice logicalDevice, const std::vector<VkDescriptorSetLayoutBinding> &setLayoutBindings = {});

VkFramebuffer createFramebuffer(VkDevice logicalDevice, VkRenderPass renderpass, const std::vector<VkImageView> &attachments, const VkExtent3D &extent);

VkPipelineLayout createPipelineLayout(VkDevice logicalDevice, const std::vector<VkDescriptorSetLayout> &setLayouts, const std::vector<VkPushConstantRange> &pushConstantRanges = {});

VkCommandPool createCommandPool(VkDevice logicalDevice, uint32_t queueFamilyIndex);

std::vector<VkCommandBuffer> createCommandBuffers(VkDevice logicalDevice, VkCommandPool commandPool, uint32_t count);

VkSemaphore createSemaphore(VkDevice logicalDevice);

VkFence createFence(VkDevice logicalDevice, VkFenceCreateFlags flags);

void chooseSwapExtent(const VkSurfaceCapabilitiesKHR &surfaceCaps, GLFWwindow *window, VkExtent2D &outExtent);

void createBuffer(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer &outBuffer, VkDeviceMemory &outBufferMemory);

uint32_t findMemoryTypeIndex(VkPhysicalDevice physicalDevice, uint32_t typeIndexFilter, VkMemoryPropertyFlags properties);

VkDeviceMemory allocateMemory(VkDevice logicalDevice, VkDeviceSize memsize, uint32_t memoryTypeIndex);

VkDescriptorPool createDescriptorPool(VkDevice logicalDevice, uint32_t maxSetCount, const std::vector<VkDescriptorPoolSize> &descriptorPoolSizes);

std::vector<VkDescriptorSet> createDescriptorSets(VkDevice logicalDevice, VkDescriptorPool descriptorPool, const std::vector<VkDescriptorSetLayout> &descriptorSetLayouts);

void createImage(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, const VkImageCreateInfo &imageCreateInfo, VkMemoryPropertyFlags properties, VkImage &outImage, VkDeviceMemory &outImageMemory);

VkCommandBuffer beginOneTimeCommands(VkDevice logicalDevice, VkCommandPool commandPool);

void endOneTimeCommands(VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool, VkCommandBuffer commandBuffer);

void copyBuffer2Image(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkImage dstImage, VkImageLayout dstImageLayout, VkExtent3D dstImageExtent);

void transitionImageLayout(VkCommandBuffer commandBuffer,
                           VkImage image,
                           VkImageLayout oldLayout,
                           VkImageLayout newLayout);

VkImageView createImageView(VkDevice logicalDevice, VkImage image, VkImageViewType imageViewType, VkFormat imageFormat, VkImageAspectFlags imageAspectMask, uint32_t mipmapLevels);

VkSampler createSampler(VkDevice logicalDevice, VkFilter magFilter, VkFilter minFilter, VkSamplerMipmapMode mipmapMode, float maxLod);

VkFormat findSupportedFormat(VkPhysicalDevice physicalDevice, const std::vector<VkFormat> &candidates, VkImageTiling tiling, VkFormatFeatureFlags features);

void generateMipmaps(VkCommandBuffer commandBuffer, VkImage image, int32_t width, int32_t height, uint32_t mipLevels, VkFilter filter);

VkSampleCountFlagBits getMaxUsableSampleCount(VkPhysicalDevice physicalDevice);