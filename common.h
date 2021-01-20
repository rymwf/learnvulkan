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

#include <glm/glm.hpp>

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

uint32_t findQueueFamilyIndexByFlag(std::vector<VkQueueFamilyProperties> &queueFamilyProperties, VkQueueFlagBits flag);
uint32_t findQueueFamilyIndexPresent(VkPhysicalDevice physicalDevice, uint32_t familyNum, VkSurfaceKHR surface);

void createImageViews(VkDevice logicalDevice,
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

void createBuffer(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,VkBuffer& outBuffer,VkDeviceMemory& outBufferMemory);

uint32_t findMemoryTypeIndex(VkPhysicalDevice physicalDevice, uint32_t typeIndexFilter, VkMemoryPropertyFlags properties);

VkDeviceMemory allocateMemory(VkDevice logicalDevice, uint32_t memoryTypeIndex, VkDeviceSize memsize);

void copyBuffer(VkDevice logicalDevice, VkCommandPool commandPool, VkQueue queue, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize bufferSize);