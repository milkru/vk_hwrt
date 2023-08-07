#include <volk.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <array>
#include <set>
#include <unordered_map>

#ifndef VK_CALL
#define VK_CALL(vkCall)                                                        \
    do                                                                         \
    {                                                                          \
        VkResult result = vkCall;                                              \
        assert(result == VK_SUCCESS);                                          \
    } while (0)
#endif // VK_CALL

const uint32_t kWidth = 1280;
const uint32_t kHeight = 720;

const char* kModelPath = "../assets/models/dragon.obj";

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
    VK_KHR_SPIRV_1_4_EXTENSION_NAME,
    VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

struct SwapchainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct Vertex
{
    float pos[3];
    float norm[3];

    bool operator==(const Vertex& other) const
    {
        return pos[0] == other.pos[0] && pos[1] == other.pos[1] &&
               pos[2] == other.pos[2];
    }
};

namespace std
{
template <> struct hash<Vertex>
{
    size_t operator()(Vertex const& vertex) const
    {
        return ((hash<float>()(vertex.pos[0]) ^
                 (hash<float>()(vertex.pos[0]) << 1)) >>
                1) ^
               (hash<float>()(vertex.pos[0]) << 1);
    }
};
}

struct UniformBufferObject
{
    glm::mat4 viewInverse;
    glm::mat4 projInverse;
    glm::vec4 lightDir;
};

struct AccelerationStructure
{
    VkAccelerationStructureKHR handle;
    uint64_t deviceAddress = 0;
    VkDeviceMemory memory;
    VkBuffer buffer;
};

struct ScratchAccelerationStructure
{
    uint64_t deviceAddress = 0;
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

struct Buffer
{
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* data = nullptr;
};

class Renderer
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
    GLFWwindow* window;

    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    VkQueue graphicsQueue;

    VkSwapchainKHR swapchain;
    std::vector<VkImage> swapchainImages;
    VkFormat swapchainImageFormat;
    VkExtent2D swapchainExtent;
    std::vector<VkImageView> swapchainImageViews;

    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;

    VkCommandPool commandPool;

    VkImage storageImage;
    VkDeviceMemory storageImageMemory;
    VkImageView storageImageView;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    Buffer vertexBuffer;
    Buffer indexBuffer;
    Buffer uniformBuffer;

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR
        rayTracingPipelineProperties{};
    AccelerationStructure blas{};
    AccelerationStructure tlas{};
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups{};

    Buffer rayGenSbtBuffer;
    Buffer rayMissSbtBuffer;
    Buffer rayHitSbtBuffer;

    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;

    VkCommandBuffer mainCommandBuffer;

    VkSemaphore imageAvailableSemaphore;
    VkSemaphore renderFinishedSemaphore;

    void initWindow()
    {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

        window =
            glfwCreateWindow(kWidth, kHeight, "Vulkan RT", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
    }

    void initVulkan()
    {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapchain();
        createSwapchainImageViews();
        createPipeline();
        createSbts();
        createCommandPool();
        loadModel();
        createVertexBuffer();
        createIndexBuffer();
        createBlas();
        createTlas();
        createStorageImage();
        createUniformBuffer();
        createDescriptorPool();
        createDescriptorSet();
        createCommandBuffer();
        createSemaphores();
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            drawFrame();
        }

        vkDeviceWaitIdle(device);
    }

    void cleanupSwapchain()
    {
        for (auto imageView : swapchainImageViews)
        {
            vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device, swapchain, nullptr);
    }

    void cleanup()
    {
        cleanupSwapchain();

        vkDestroyImageView(device, storageImageView, nullptr);
        vkDestroyImage(device, storageImage, nullptr);
        vkFreeMemory(device, storageImageMemory, nullptr);

        destroyAccelerationStructureBuffer(tlas);
        destroyAccelerationStructureBuffer(blas);

        destroyBuffer(rayHitSbtBuffer);
        destroyBuffer(rayMissSbtBuffer);
        destroyBuffer(rayGenSbtBuffer);

        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

        destroyBuffer(uniformBuffer);

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        destroyBuffer(indexBuffer);
        destroyBuffer(vertexBuffer);

        vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);
        vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);

        vkDestroyCommandPool(device, commandPool, nullptr);

        vkDestroyDevice(device, nullptr);

        if (enableValidationLayers)
        {
            vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);

        glfwTerminate();
    }

    void createInstance()
    {
        VK_CALL(volkInitialize());
        assert(!enableValidationLayers || checkValidationLayerSupport());

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_1;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = uint32_t(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers)
        {
            createInfo.enabledLayerCount = uint32_t(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext =
                (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
        }
        else
        {
            createInfo.enabledLayerCount = 0;

            createInfo.pNext = nullptr;
        }

        VK_CALL(vkCreateInstance(&createInfo, nullptr, &instance));

        volkLoadInstance(instance);
    }

    void populateDebugMessengerCreateInfo(
        VkDebugUtilsMessengerCreateInfoEXT& createInfo)
    {
        createInfo = {};
        createInfo.sType =
            VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }

    void setupDebugMessenger()
    {
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        VK_CALL(vkCreateDebugUtilsMessengerEXT(
            instance, &createInfo, nullptr, &debugMessenger));
    }

    void createSurface()
    {
        VK_CALL(glfwCreateWindowSurface(instance, window, nullptr, &surface));
    }

    void pickPhysicalDevice()
    {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        assert(deviceCount > 0);

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto& device : devices)
        {
            if (isDeviceSuitable(device))
            {
                physicalDevice = device;
                break;
            }
        }

        assert(physicalDevice != VK_NULL_HANDLE);
    }

    void createLogicalDevice()
    {
        uint32_t graphicsFamily = tryFindGraphicsFamily();
        assert(graphicsFamily != -1);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = graphicsFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = uint32_t(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.enabledExtensionCount = uint32_t(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (enableValidationLayers)
        {
            createInfo.enabledLayerCount = uint32_t(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else
        {
            createInfo.enabledLayerCount = 0;
        }

        VkPhysicalDeviceBufferDeviceAddressFeatures deviceAddresFeatures{};
        deviceAddresFeatures.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
        deviceAddresFeatures.bufferDeviceAddress = VK_TRUE;

        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingFeatures{};
        rayTracingFeatures.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
        rayTracingFeatures.rayTracingPipeline = VK_TRUE;
        rayTracingFeatures.pNext = &deviceAddresFeatures;

        VkPhysicalDeviceAccelerationStructureFeaturesKHR
            accelerationStructureFeatures{};
        accelerationStructureFeatures.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
        accelerationStructureFeatures.accelerationStructure = VK_TRUE;
        accelerationStructureFeatures.pNext = &rayTracingFeatures;

        VkPhysicalDeviceFeatures2 deviceFeatures2{};
        deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        deviceFeatures2.pNext = &accelerationStructureFeatures;

        createInfo.pNext = &deviceFeatures2;

        VK_CALL(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device));

        vkGetDeviceQueue(device, graphicsFamily, 0, &graphicsQueue);

        rayTracingPipelineProperties.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
        rayTracingPipelineProperties.pNext = nullptr;

        VkPhysicalDeviceProperties2 deviceProperties2{};
        deviceProperties2.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        deviceProperties2.pNext = &rayTracingPipelineProperties;
        vkGetPhysicalDeviceProperties2(physicalDevice, &deviceProperties2);
    }

    void createSwapchain()
    {
        SwapchainSupportDetails swapChainSupport =
            querySwapchainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat =
            tryChooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode =
            chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 &&
            imageCount > swapChainSupport.capabilities.maxImageCount)
        {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;

        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.preTransform =
            swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;

        VK_CALL(vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain));

        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
        swapchainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(
            device, swapchain, &imageCount, swapchainImages.data());

        swapchainImageFormat = surfaceFormat.format;
        swapchainExtent = extent;
    }

    void createSwapchainImageViews()
    {
        swapchainImageViews.resize(swapchainImages.size());

        for (uint32_t i = 0; i < swapchainImages.size(); i++)
        {
            swapchainImageViews[i] = createImageView(swapchainImages[i],
                                                     swapchainImageFormat,
                                                     VK_IMAGE_ASPECT_COLOR_BIT);
        }
    }

    void createCommandPool()
    {
        uint32_t graphicsFamily = tryFindGraphicsFamily();
        assert(graphicsFamily != -1);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = graphicsFamily;

        VK_CALL(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool));
    }

    VkImageView createImageView(VkImage image,
                                VkFormat format,
                                VkImageAspectFlags aspectFlags)
    {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspectFlags;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        VkImageView imageView;
        VK_CALL(vkCreateImageView(device, &viewInfo, nullptr, &imageView));

        return imageView;
    }

    void createStorageImage()
    {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = kWidth;
        imageInfo.extent.height = kHeight;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.format = swapchainImageFormat;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage =
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VK_CALL(vkCreateImage(device, &imageInfo, nullptr, &storageImage));

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, storageImage, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex =
            tryFindMemoryType(memRequirements.memoryTypeBits,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        VK_CALL(
            vkAllocateMemory(device, &allocInfo, nullptr, &storageImageMemory));

        VK_CALL(vkBindImageMemory(device, storageImage, storageImageMemory, 0));

        storageImageView = createImageView(
            storageImage, swapchainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);

        transitionImageLayout(storageImage,
                              VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_GENERAL,
                              {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    }

    void transitionImageLayout(
        VkCommandBuffer cmdbuffer,
        VkImage image,
        VkImageLayout oldImageLayout,
        VkImageLayout newImageLayout,
        VkImageSubresourceRange subresourceRange,
        VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)
    {
        VkImageMemoryBarrier imageMemoryBarrier{};
        imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imageMemoryBarrier.oldLayout = oldImageLayout;
        imageMemoryBarrier.newLayout = newImageLayout;
        imageMemoryBarrier.image = image;
        imageMemoryBarrier.subresourceRange = subresourceRange;

        switch (oldImageLayout)
        {
        case VK_IMAGE_LAYOUT_UNDEFINED:
            imageMemoryBarrier.srcAccessMask = 0;
            break;

        case VK_IMAGE_LAYOUT_PREINITIALIZED:
            imageMemoryBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            imageMemoryBarrier.srcAccessMask =
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            imageMemoryBarrier.srcAccessMask =
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            break;

        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            break;
        default:
            break;
        }

        switch (newImageLayout)
        {
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            break;

        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            imageMemoryBarrier.dstAccessMask =
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            imageMemoryBarrier.dstAccessMask =
                imageMemoryBarrier.dstAccessMask |
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            if (imageMemoryBarrier.srcAccessMask == 0)
            {
                imageMemoryBarrier.srcAccessMask =
                    VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            }
            imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            break;
        default:
            break;
        }

        vkCmdPipelineBarrier(cmdbuffer,
                             srcStageMask,
                             dstStageMask,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &imageMemoryBarrier);
    }

    void transitionImageLayout(VkImage image,
                               VkImageLayout oldLayout,
                               VkImageLayout newLayout,
                               VkImageSubresourceRange subresourceRange)
    {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        transitionImageLayout(
            commandBuffer, image, oldLayout, newLayout, subresourceRange);

        endSingleTimeCommands(commandBuffer);
    }

    uint64_t getBufferDeviceAddress(Buffer& buffer)
    {
        VkBufferDeviceAddressInfoKHR bufferDeviceAddressInfo{};
        bufferDeviceAddressInfo.sType =
            VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        bufferDeviceAddressInfo.buffer = buffer.handle;

        return vkGetBufferDeviceAddressKHR(device, &bufferDeviceAddressInfo);
    }

    void createAccelerationStructureBuffer(
        AccelerationStructure& accelerationStructure,
        VkAccelerationStructureBuildSizesInfoKHR buildSizeInfo)
    {
        VkBufferCreateInfo bufferCreateInfo{};
        bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.size = buildSizeInfo.accelerationStructureSize;
        bufferCreateInfo.usage =
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

        vkCreateBuffer(
            device, &bufferCreateInfo, nullptr, &accelerationStructure.buffer);

        VkMemoryRequirements memoryRequirements{};
        vkGetBufferMemoryRequirements(
            device, accelerationStructure.buffer, &memoryRequirements);

        VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo{};
        memoryAllocateFlagsInfo.sType =
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        memoryAllocateFlagsInfo.flags =
            VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;

        VkMemoryAllocateInfo memoryAllocateInfo{};
        memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        memoryAllocateInfo.pNext = &memoryAllocateFlagsInfo;
        memoryAllocateInfo.allocationSize = memoryRequirements.size;
        memoryAllocateInfo.memoryTypeIndex =
            tryFindMemoryType(memoryRequirements.memoryTypeBits,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        vkAllocateMemory(device,
                         &memoryAllocateInfo,
                         nullptr,
                         &accelerationStructure.memory);

        vkBindBufferMemory(device,
                           accelerationStructure.buffer,
                           accelerationStructure.memory,
                           0);
    }

    void destroyAccelerationStructureBuffer(
        AccelerationStructure& accelerationStructure)
    {
        if (accelerationStructure.handle != VK_NULL_HANDLE)
        {
            vkDestroyAccelerationStructureKHR(
                device, accelerationStructure.handle, nullptr);
        }

        if (accelerationStructure.buffer != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(device, accelerationStructure.buffer, nullptr);
        }

        if (accelerationStructure.memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(device, accelerationStructure.memory, nullptr);
        }
    }

    ScratchAccelerationStructure createScratchBufferAs(VkDeviceSize size)
    {
        ScratchAccelerationStructure scratchBuffer{};

        VkBufferCreateInfo bufferCreateInfo{};
        bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.size = size;
        bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        VK_CALL(vkCreateBuffer(
            device, &bufferCreateInfo, nullptr, &scratchBuffer.handle));

        VkMemoryRequirements memoryRequirements{};
        vkGetBufferMemoryRequirements(
            device, scratchBuffer.handle, &memoryRequirements);

        VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo{};
        memoryAllocateFlagsInfo.sType =
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        memoryAllocateFlagsInfo.flags =
            VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;

        VkMemoryAllocateInfo memoryAllocateInfo = {};
        memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        memoryAllocateInfo.pNext = &memoryAllocateFlagsInfo;
        memoryAllocateInfo.allocationSize = memoryRequirements.size;
        memoryAllocateInfo.memoryTypeIndex =
            tryFindMemoryType(memoryRequirements.memoryTypeBits,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CALL(vkAllocateMemory(
            device, &memoryAllocateInfo, nullptr, &scratchBuffer.memory));
        VK_CALL(vkBindBufferMemory(
            device, scratchBuffer.handle, scratchBuffer.memory, 0));

        VkBufferDeviceAddressInfoKHR bufferDeviceAddressInfo{};
        bufferDeviceAddressInfo.sType =
            VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        bufferDeviceAddressInfo.buffer = scratchBuffer.handle;
        scratchBuffer.deviceAddress =
            vkGetBufferDeviceAddressKHR(device, &bufferDeviceAddressInfo);

        return scratchBuffer;
    }

    void deleteScratchBufferAs(ScratchAccelerationStructure& scratchBuffer)
    {
        if (scratchBuffer.memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(device, scratchBuffer.memory, nullptr);
        }

        if (scratchBuffer.handle != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(device, scratchBuffer.handle, nullptr);
        }
    }

    void createBlas()
    {
        uint32_t indexCount = uint32_t(indices.size());
        uint32_t triangleCount = indexCount / 3;

        VkAccelerationStructureGeometryKHR geometry{};
        geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometry.geometry.triangles.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        geometry.geometry.triangles.vertexData.deviceAddress =
            getBufferDeviceAddress(vertexBuffer);
        geometry.geometry.triangles.maxVertex = uint32_t(vertices.size());
        geometry.geometry.triangles.vertexStride = sizeof(Vertex);
        geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
        geometry.geometry.triangles.indexData.deviceAddress =
            getBufferDeviceAddress(indexBuffer);

        VkAccelerationStructureBuildGeometryInfoKHR buildGeometryInfo{};
        buildGeometryInfo.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        buildGeometryInfo.type =
            VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        buildGeometryInfo.flags =
            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        buildGeometryInfo.geometryCount = 1;
        buildGeometryInfo.pGeometries = &geometry;

        uint32_t numTriangles = indices.size() / 3;
        VkAccelerationStructureBuildSizesInfoKHR buildSizesInfo{};
        buildSizesInfo.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        vkGetAccelerationStructureBuildSizesKHR(
            device,
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &buildGeometryInfo,
            &numTriangles,
            &buildSizesInfo);

        createAccelerationStructureBuffer(blas, buildSizesInfo);

        VkAccelerationStructureCreateInfoKHR accelerationStructureCreateInfo{};
        accelerationStructureCreateInfo.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        accelerationStructureCreateInfo.buffer = blas.buffer;
        accelerationStructureCreateInfo.size =
            buildSizesInfo.accelerationStructureSize;
        accelerationStructureCreateInfo.type =
            VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        vkCreateAccelerationStructureKHR(
            device, &accelerationStructureCreateInfo, nullptr, &blas.handle);

        ScratchAccelerationStructure scratchBuffer =
            createScratchBufferAs(buildSizesInfo.buildScratchSize);

        buildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        buildGeometryInfo.dstAccelerationStructure = blas.handle;
        buildGeometryInfo.scratchData.deviceAddress =
            scratchBuffer.deviceAddress;

        VkAccelerationStructureBuildRangeInfoKHR buildRangeInfo{};
        buildRangeInfo.primitiveCount = numTriangles;
        buildRangeInfo.primitiveOffset = 0;
        buildRangeInfo.firstVertex = 0;
        buildRangeInfo.transformOffset = 0;

        std::vector<VkAccelerationStructureBuildRangeInfoKHR*>
            accelerationBuildStructureRangeInfos = {&buildRangeInfo};

        {
            VkCommandBuffer commandBuffer = beginSingleTimeCommands();

            vkCmdBuildAccelerationStructuresKHR(
                commandBuffer,
                1,
                &buildGeometryInfo,
                accelerationBuildStructureRangeInfos.data());

            endSingleTimeCommands(commandBuffer);
        }

        VkAccelerationStructureDeviceAddressInfoKHR deviceAddressInfo{};
        deviceAddressInfo.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        deviceAddressInfo.accelerationStructure = blas.handle;
        blas.deviceAddress = vkGetAccelerationStructureDeviceAddressKHR(
            device, &deviceAddressInfo);

        deleteScratchBufferAs(scratchBuffer);
    }

    void createTlas()
    {
        VkTransformMatrixKHR transformMatrix = {1.0f,
                                                0.0f,
                                                0.0f,
                                                0.0f,
                                                0.0f,
                                                1.0f,
                                                0.0f,
                                                0.0f,
                                                0.0f,
                                                0.0f,
                                                1.0f,
                                                0.0f};

        VkAccelerationStructureInstanceKHR instance{};
        instance.transform = transformMatrix;
        instance.instanceCustomIndex = 0;
        instance.mask = 0xFF;
        instance.instanceShaderBindingTableRecordOffset = 0;
        instance.flags =
            VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        instance.accelerationStructureReference = blas.deviceAddress;

        Buffer instancesBuffer;
        createBuffer(
            sizeof(VkAccelerationStructureInstanceKHR),
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            instancesBuffer,
            &instance);

        VkAccelerationStructureGeometryKHR geometry{};
        geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        geometry.geometry.instances.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        geometry.geometry.instances.arrayOfPointers = VK_FALSE;
        geometry.geometry.instances.data.deviceAddress =
            getBufferDeviceAddress(instancesBuffer);

        VkAccelerationStructureBuildGeometryInfoKHR buildGeometryInfo{};
        buildGeometryInfo.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        buildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        buildGeometryInfo.flags =
            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        buildGeometryInfo.geometryCount = 1;
        buildGeometryInfo.pGeometries = &geometry;

        uint32_t triangleCount = indices.size() / 3;
        VkAccelerationStructureBuildSizesInfoKHR
            accelerationStructureBuildSizesInfo{};
        accelerationStructureBuildSizesInfo.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        vkGetAccelerationStructureBuildSizesKHR(
            device,
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &buildGeometryInfo,
            &triangleCount,
            &accelerationStructureBuildSizesInfo);

        createAccelerationStructureBuffer(tlas,
                                          accelerationStructureBuildSizesInfo);

        VkAccelerationStructureCreateInfoKHR accelerationStructureCreateInfo{};
        accelerationStructureCreateInfo.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        accelerationStructureCreateInfo.buffer = tlas.buffer;
        accelerationStructureCreateInfo.size =
            accelerationStructureBuildSizesInfo.accelerationStructureSize;
        accelerationStructureCreateInfo.type =
            VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        vkCreateAccelerationStructureKHR(
            device, &accelerationStructureCreateInfo, nullptr, &tlas.handle);

        ScratchAccelerationStructure scratchBuffer = createScratchBufferAs(
            accelerationStructureBuildSizesInfo.buildScratchSize);

        buildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        buildGeometryInfo.dstAccelerationStructure = tlas.handle;
        buildGeometryInfo.geometryCount = 1;
        buildGeometryInfo.pGeometries = &geometry;
        buildGeometryInfo.scratchData.deviceAddress =
            scratchBuffer.deviceAddress;

        VkAccelerationStructureBuildRangeInfoKHR buildRangeInfo{};
        buildRangeInfo.primitiveCount = 1;
        buildRangeInfo.primitiveOffset = 0;
        buildRangeInfo.firstVertex = 0;
        buildRangeInfo.transformOffset = 0;

        std::vector<VkAccelerationStructureBuildRangeInfoKHR*>
            accelerationBuildStructureRangeInfos = {&buildRangeInfo};

        {
            VkCommandBuffer commandBuffer = beginSingleTimeCommands();

            vkCmdBuildAccelerationStructuresKHR(
                commandBuffer,
                1,
                &buildGeometryInfo,
                accelerationBuildStructureRangeInfos.data());

            endSingleTimeCommands(commandBuffer);
        }

        VkAccelerationStructureDeviceAddressInfoKHR deviceAddressInfo{};
        deviceAddressInfo.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        deviceAddressInfo.accelerationStructure = tlas.handle;
        tlas.deviceAddress = vkGetAccelerationStructureDeviceAddressKHR(
            device, &deviceAddressInfo);

        deleteScratchBufferAs(scratchBuffer);
        destroyBuffer(instancesBuffer);
    }

    void createPipeline()
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings;

        { // TLAS
            VkDescriptorSetLayoutBinding binding{};
            binding.binding = 0;
            binding.descriptorCount = 1;
            binding.descriptorType =
                VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            binding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                 VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            bindings.push_back(binding);
        }

        { // Target
            VkDescriptorSetLayoutBinding binding{};
            binding.binding = 1;
            binding.descriptorCount = 1;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            binding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            bindings.push_back(binding);
        }

        { // UBO
            VkDescriptorSetLayoutBinding binding{};
            binding.binding = 2;
            binding.descriptorCount = 1;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            binding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                 VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            bindings.push_back(binding);
        }

        { // Vertex buffer
            VkDescriptorSetLayoutBinding binding{};
            binding.binding = 3;
            binding.descriptorCount = 1;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binding.stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            bindings.push_back(binding);
        }

        { // Index buffer
            VkDescriptorSetLayoutBinding binding{};
            binding.binding = 4;
            binding.descriptorCount = 1;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binding.stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            bindings.push_back(binding);
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = uint32_t(bindings.size());
        layoutInfo.pBindings = bindings.data();

        VK_CALL(vkCreateDescriptorSetLayout(
            device, &layoutInfo, nullptr, &descriptorSetLayout));

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType =
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

        VK_CALL(vkCreatePipelineLayout(
            device, &pipelineLayoutInfo, nullptr, &pipelineLayout));

        std::vector<VkPipelineShaderStageCreateInfo> shaderStages;

        { // Ray generation shader
            auto shaderCode = readFile("shaders/geometry.rgen.spv");
            VkShaderModule shaderModule = createShaderModule(shaderCode);

            VkRayTracingShaderGroupCreateInfoKHR shaderGroup{};
            shaderGroup.sType =
                VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
            shaderGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
            shaderGroup.generalShader = uint32_t(shaderStages.size());
            shaderGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
            shaderGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
            shaderGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
            shaderGroups.push_back(shaderGroup);

            VkPipelineShaderStageCreateInfo shaderStageInfo{};
            shaderStageInfo.sType =
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            shaderStageInfo.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            shaderStageInfo.module = shaderModule;
            shaderStageInfo.pName = "main";
            shaderStages.push_back(shaderStageInfo);
        }

        { // Miss shader
            auto shaderCode = readFile("shaders/geometry.rmiss.spv");
            VkShaderModule shaderModule = createShaderModule(shaderCode);

            VkRayTracingShaderGroupCreateInfoKHR shaderGroup{};
            shaderGroup.sType =
                VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
            shaderGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
            shaderGroup.generalShader = uint32_t(shaderStages.size());
            shaderGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
            shaderGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
            shaderGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
            shaderGroups.push_back(shaderGroup);

            VkPipelineShaderStageCreateInfo shaderStageInfo{};
            shaderStageInfo.sType =
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            shaderStageInfo.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
            shaderStageInfo.module = shaderModule;
            shaderStageInfo.pName = "main";
            shaderStages.push_back(shaderStageInfo);
        }

        { // Shadow miss shader
            auto shaderCode = readFile("shaders/shadow.rmiss.spv");
            VkShaderModule shaderModule = createShaderModule(shaderCode);

            VkRayTracingShaderGroupCreateInfoKHR shaderGroup{};
            shaderGroup.sType =
                VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
            shaderGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
            shaderGroup.generalShader = uint32_t(shaderStages.size());
            shaderGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
            shaderGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
            shaderGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
            shaderGroups.push_back(shaderGroup);

            VkPipelineShaderStageCreateInfo shaderStageInfo{};
            shaderStageInfo.sType =
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            shaderStageInfo.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
            shaderStageInfo.module = shaderModule;
            shaderStageInfo.pName = "main";
            shaderStages.push_back(shaderStageInfo);
        }

        { // Closest hit shader
            auto shaderCode = readFile("shaders/geometry.rchit.spv");
            VkShaderModule shaderModule = createShaderModule(shaderCode);

            VkRayTracingShaderGroupCreateInfoKHR shaderGroup{};
            shaderGroup.sType =
                VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
            shaderGroup.type =
                VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
            shaderGroup.generalShader = VK_SHADER_UNUSED_KHR;
            shaderGroup.closestHitShader = uint32_t(shaderStages.size());
            shaderGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
            shaderGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
            shaderGroups.push_back(shaderGroup);

            VkPipelineShaderStageCreateInfo shaderStageInfo{};
            shaderStageInfo.sType =
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            shaderStageInfo.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            shaderStageInfo.module = shaderModule;
            shaderStageInfo.pName = "main";
            shaderStages.push_back(shaderStageInfo);
        }

        VkRayTracingPipelineCreateInfoKHR pipelineInfo{};
        pipelineInfo.sType =
            VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
        pipelineInfo.stageCount = uint32_t(shaderStages.size());
        pipelineInfo.pStages = shaderStages.data();
        pipelineInfo.groupCount = uint32_t(shaderGroups.size());
        pipelineInfo.pGroups = shaderGroups.data();
        pipelineInfo.maxPipelineRayRecursionDepth = 1;
        pipelineInfo.layout = pipelineLayout;

        VK_CALL(vkCreateRayTracingPipelinesKHR(device,
                                               VK_NULL_HANDLE,
                                               VK_NULL_HANDLE,
                                               1,
                                               &pipelineInfo,
                                               nullptr,
                                               &pipeline));

        for (VkPipelineShaderStageCreateInfo& shaderStage : shaderStages)
        {
            vkDestroyShaderModule(device, shaderStage.module, nullptr);
        }
    }

    uint32_t getAlignedSize(uint32_t value, uint32_t alignment)
    {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    void createSbts()
    {
        const uint32_t handleSize =
            rayTracingPipelineProperties.shaderGroupHandleSize;
        const uint32_t handleSizeAligned = getAlignedSize(
            rayTracingPipelineProperties.shaderGroupHandleSize,
            rayTracingPipelineProperties.shaderGroupHandleAlignment);
        const uint32_t groupCount = static_cast<uint32_t>(shaderGroups.size());
        const uint32_t sbtSize = groupCount * handleSizeAligned;

        std::vector<uint8_t> shaderHandleStorage(sbtSize);

        vkGetRayTracingShaderGroupHandlesKHR(device,
                                             pipeline,
                                             0,
                                             groupCount,
                                             sbtSize,
                                             shaderHandleStorage.data());

        const VkBufferUsageFlags bufferUsageFlags =
            VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

        const VkMemoryPropertyFlags memoryUsageFlags =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        createBuffer(handleSize,
                     bufferUsageFlags,
                     memoryUsageFlags,
                     rayGenSbtBuffer,
                     shaderHandleStorage.data());

        createBuffer(2 * size_t(handleSize),
                     bufferUsageFlags,
                     memoryUsageFlags,
                     rayMissSbtBuffer,
                     shaderHandleStorage.data() + handleSizeAligned);

        createBuffer(handleSize,
                     bufferUsageFlags,
                     memoryUsageFlags,
                     rayHitSbtBuffer,
                     shaderHandleStorage.data() +
                         3 * size_t(handleSizeAligned));
    }

    void loadModel()
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        bool result = tinyobj::LoadObj(
            &attrib, &shapes, &materials, &warn, &err, kModelPath);
        assert(result);

        std::unordered_map<Vertex, uint32_t> uniqueVertices{};

        for (const auto& shape : shapes)
        {
            for (const auto& index : shape.mesh.indices)
            {
                Vertex vertex{};

                vertex.pos[0] =
                    attrib.vertices[3 * size_t(index.vertex_index) + 0];
                vertex.pos[1] =
                    attrib.vertices[3 * size_t(index.vertex_index) + 1];
                vertex.pos[2] =
                    attrib.vertices[3 * size_t(index.vertex_index) + 2];

                vertex.norm[0] =
                    attrib.normals[3 * size_t(index.normal_index) + 0];
                vertex.norm[1] =
                    attrib.normals[3 * size_t(index.normal_index) + 1];
                vertex.norm[2] =
                    attrib.normals[3 * size_t(index.normal_index) + 2];

                if (uniqueVertices.count(vertex) == 0)
                {
                    uniqueVertices[vertex] =
                        static_cast<uint32_t>(vertices.size());
                    vertices.push_back(vertex);
                }

                indices.push_back(uniqueVertices[vertex]);
            }
        }
    }

    void createVertexBuffer()
    {
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer,
                     stagingBufferMemory,
                     vertices.data());

        createBuffer(
            bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            vertexBuffer);

        copyBuffer(stagingBuffer, vertexBuffer.handle, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createIndexBuffer()
    {
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer,
                     stagingBufferMemory,
                     indices.data());

        createBuffer(
            bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            indexBuffer);

        copyBuffer(stagingBuffer, indexBuffer.handle, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createUniformBuffer()
    {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        createBuffer(bufferSize,
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     uniformBuffer);

        vkMapMemory(device,
                    uniformBuffer.memory,
                    0,
                    bufferSize,
                    0,
                    &uniformBuffer.data);
    }

    void createDescriptorPool()
    {
        std::array<VkDescriptorPoolSize, 4> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        poolSizes[0].descriptorCount = 1;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        poolSizes[1].descriptorCount = 1;
        poolSizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[2].descriptorCount = 1;
        poolSizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[3].descriptorCount = 2;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = uint32_t(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = 1;

        VK_CALL(vkCreateDescriptorPool(
            device, &poolInfo, nullptr, &descriptorPool));
    }

    void createDescriptorSet()
    {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;

        VK_CALL(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));

        std::array<VkWriteDescriptorSet, 5> descriptorWrites{};

        VkWriteDescriptorSetAccelerationStructureKHR descriptorASInfo{};
        descriptorASInfo.sType =
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        descriptorASInfo.accelerationStructureCount = 1;
        descriptorASInfo.pAccelerationStructures = &tlas.handle;

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].pNext = &descriptorASInfo;
        descriptorWrites[0].dstSet = descriptorSet;
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType =
            VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        descriptorWrites[0].descriptorCount = 1;

        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageInfo.imageView = storageImageView;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = descriptorSet;
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &imageInfo;

        VkDescriptorBufferInfo uniformBufferInfo{};
        uniformBufferInfo.buffer = uniformBuffer.handle;
        uniformBufferInfo.offset = 0;
        uniformBufferInfo.range = sizeof(UniformBufferObject);

        descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[2].dstSet = descriptorSet;
        descriptorWrites[2].dstBinding = 2;
        descriptorWrites[2].dstArrayElement = 0;
        descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[2].descriptorCount = 1;
        descriptorWrites[2].pBufferInfo = &uniformBufferInfo;

        VkDescriptorBufferInfo vertexBufferInfo{};
        vertexBufferInfo.buffer = vertexBuffer.handle;
        vertexBufferInfo.offset = 0;
        vertexBufferInfo.range = vertices.size() * sizeof(Vertex);

        descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[3].dstSet = descriptorSet;
        descriptorWrites[3].dstBinding = 3;
        descriptorWrites[3].dstArrayElement = 0;
        descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[3].descriptorCount = 1;
        descriptorWrites[3].pBufferInfo = &vertexBufferInfo;

        VkDescriptorBufferInfo indexBufferInfo{};
        indexBufferInfo.buffer = indexBuffer.handle;
        indexBufferInfo.offset = 0;
        indexBufferInfo.range = indices.size() * sizeof(uint32_t);

        descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[4].dstSet = descriptorSet;
        descriptorWrites[4].dstBinding = 4;
        descriptorWrites[4].dstArrayElement = 0;
        descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[4].descriptorCount = 1;
        descriptorWrites[4].pBufferInfo = &indexBufferInfo;

        vkUpdateDescriptorSets(device,
                               uint32_t(descriptorWrites.size()),
                               descriptorWrites.data(),
                               0,
                               nullptr);
    }

    void createBuffer(VkDeviceSize size,
                      VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties,
                      VkBuffer& buffer,
                      VkDeviceMemory& bufferMemory,
                      void* data = nullptr)
    {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VK_CALL(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer));

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex =
            tryFindMemoryType(memRequirements.memoryTypeBits, properties);

        VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo{};
        memoryAllocateFlagsInfo.sType =
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        memoryAllocateFlagsInfo.flags =
            VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;

        if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
        {
            allocInfo.pNext = &memoryAllocateFlagsInfo;
        }

        VK_CALL(vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory));

        vkBindBufferMemory(device, buffer, bufferMemory, 0);

        if (data != nullptr)
        {
            void* mapped;
            vkMapMemory(device, bufferMemory, 0, size, 0, &mapped);
            memcpy(mapped, data, size);
        }
    }

    void createBuffer(VkDeviceSize size,
                      VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties,
                      Buffer& buffer,
                      void* data = nullptr)
    {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VK_CALL(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer.handle));

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer.handle, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex =
            tryFindMemoryType(memRequirements.memoryTypeBits, properties);

        VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo{};
        memoryAllocateFlagsInfo.sType =
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        memoryAllocateFlagsInfo.flags =
            VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;

        if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
        {
            allocInfo.pNext = &memoryAllocateFlagsInfo;
        }

        VK_CALL(vkAllocateMemory(device, &allocInfo, nullptr, &buffer.memory));

        vkBindBufferMemory(device, buffer.handle, buffer.memory, 0);

        if (data != nullptr)
        {
            void* mapped;
            vkMapMemory(device, buffer.memory, 0, size, 0, &mapped);
            memcpy(mapped, data, size);
        }
    }

    void destroyBuffer(Buffer& buffer)
    {
        if (buffer.handle != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(device, buffer.handle, nullptr);
        }

        if (buffer.memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(device, buffer.memory, nullptr);
        }
    }

    VkCommandBuffer beginSingleTimeCommands()
    {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer)
    {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
    {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommands(commandBuffer);
    }

    uint32_t tryFindMemoryType(uint32_t typeFilter,
                               VkMemoryPropertyFlags properties)
    {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
        {
            if ((typeFilter & (1 << i)) &&
                (memProperties.memoryTypes[i].propertyFlags & properties) ==
                    properties)
            {
                return i;
            }
        }

        assert(!"Failed to find suitable memory type!");
        return -1;
    }

    void createCommandBuffer()
    {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        VK_CALL(
            vkAllocateCommandBuffers(device, &allocInfo, &mainCommandBuffer));
    }

    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        VK_CALL(vkBeginCommandBuffer(commandBuffer, &beginInfo));

        {
            uint32_t handleSizeAligned = getAlignedSize(
                rayTracingPipelineProperties.shaderGroupHandleSize,
                rayTracingPipelineProperties.shaderGroupHandleAlignment);

            VkStridedDeviceAddressRegionKHR raygenShaderSbtEntry{};
            raygenShaderSbtEntry.deviceAddress =
                getBufferDeviceAddress(rayGenSbtBuffer);
            raygenShaderSbtEntry.stride = handleSizeAligned;
            raygenShaderSbtEntry.size = handleSizeAligned;

            VkStridedDeviceAddressRegionKHR missShaderSbtEntry{};
            missShaderSbtEntry.deviceAddress =
                getBufferDeviceAddress(rayMissSbtBuffer);
            missShaderSbtEntry.stride = handleSizeAligned;
            missShaderSbtEntry.size = handleSizeAligned;

            VkStridedDeviceAddressRegionKHR hitShaderSbtEntry{};
            hitShaderSbtEntry.deviceAddress =
                getBufferDeviceAddress(rayHitSbtBuffer);
            hitShaderSbtEntry.stride = handleSizeAligned;
            hitShaderSbtEntry.size = handleSizeAligned;

            VkStridedDeviceAddressRegionKHR callableShaderSbtEntry{};

            vkCmdBindPipeline(commandBuffer,
                              VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                              pipeline);

            vkCmdBindDescriptorSets(commandBuffer,
                                    VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                    pipelineLayout,
                                    0,
                                    1,
                                    &descriptorSet,
                                    0,
                                    0);

            vkCmdTraceRaysKHR(commandBuffer,
                              &raygenShaderSbtEntry,
                              &missShaderSbtEntry,
                              &hitShaderSbtEntry,
                              &callableShaderSbtEntry,
                              swapchainExtent.width,
                              swapchainExtent.height,
                              1);

            VkImageSubresourceRange subresourceRange = {
                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

            transitionImageLayout(commandBuffer,
                                  swapchainImages[imageIndex],
                                  VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  subresourceRange);

            transitionImageLayout(commandBuffer,
                                  storageImage,
                                  VK_IMAGE_LAYOUT_GENERAL,
                                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  subresourceRange);

            VkImageCopy copyRegion{};
            copyRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copyRegion.srcOffset = {0, 0, 0};
            copyRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copyRegion.dstOffset = {0, 0, 0};
            copyRegion.extent = {kWidth, kHeight, 1};

            vkCmdCopyImage(commandBuffer,
                           storageImage,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           swapchainImages[imageIndex],
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1,
                           &copyRegion);

            transitionImageLayout(commandBuffer,
                                  swapchainImages[imageIndex],
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                  subresourceRange);

            transitionImageLayout(commandBuffer,
                                  storageImage,
                                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  VK_IMAGE_LAYOUT_GENERAL,
                                  subresourceRange);
        }

        VK_CALL(vkEndCommandBuffer(commandBuffer));
    }

    void createSemaphores()
    {
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VK_CALL(vkCreateSemaphore(
            device, &semaphoreInfo, nullptr, &imageAvailableSemaphore));

        VK_CALL(vkCreateSemaphore(
            device, &semaphoreInfo, nullptr, &renderFinishedSemaphore));
    }

    void updateUniformBuffer()
    {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(
                         currentTime - startTime)
                         .count();

        UniformBufferObject ubo{};
        ubo.viewInverse =
            glm::inverse(glm::lookAt(glm::vec3(-1.2f, 0.5f, -0.7f),
                                     glm::vec3(0.0f, 0.0f, 0.0f),
                                     glm::vec3(0.0f, 1.0f, 0.0f)));

        ubo.projInverse = glm::inverse(glm::perspective(
            glm::radians(45.0f),
            swapchainExtent.width / (float)swapchainExtent.height,
            0.1f,
            10.0f));

        glm::mat4 rotationMatrix = glm::rotate(glm::mat4(1.0f),
                                               time * glm::radians(90.0f),
                                               glm::vec3(0.0f, 1.0f, 0.0f));

        ubo.lightDir =
            glm::normalize(rotationMatrix * glm::vec4(0.0f, 0.0f, -1.0f, 0.0f));

        memcpy(uniformBuffer.data, &ubo, sizeof(ubo));
    }

    void drawFrame()
    {
        vkDeviceWaitIdle(device);

        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device,
                                                swapchain,
                                                UINT64_MAX,
                                                imageAvailableSemaphore,
                                                VK_NULL_HANDLE,
                                                &imageIndex);

        assert(result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR);

        updateUniformBuffer();

        vkResetCommandBuffer(mainCommandBuffer, 0);
        recordCommandBuffer(mainCommandBuffer, imageIndex);

        VkPipelineStageFlags waitStages[] = {
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &imageAvailableSemaphore;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &mainCommandBuffer;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &renderFinishedSemaphore;

        VK_CALL(vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE));

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &renderFinishedSemaphore;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain;
        presentInfo.pImageIndices = &imageIndex;

        VK_CALL(vkQueuePresentKHR(graphicsQueue, &presentInfo));
    }

    VkShaderModule createShaderModule(const std::vector<char>& code)
    {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        VK_CALL(
            vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule));

        return shaderModule;
    }

    VkSurfaceFormatKHR tryChooseSwapSurfaceFormat(
        const std::vector<VkSurfaceFormatKHR>& availableFormats)
    {
        for (const auto& availableFormat : availableFormats)
        {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM)
            {
                return availableFormat;
            }
        }

        assert(!"preferred surface format not found!");
        return availableFormats[0];
    }

    VkPresentModeKHR chooseSwapPresentMode(
        const std::vector<VkPresentModeKHR>& availablePresentModes)
    {
        for (const auto& availablePresentMode : availablePresentModes)
        {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
            {
                return availablePresentMode;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
    {
        if (capabilities.currentExtent.width !=
            std::numeric_limits<uint32_t>::max())
        {
            return capabilities.currentExtent;
        }
        else
        {
            int32_t width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {static_cast<uint32_t>(width),
                                       static_cast<uint32_t>(height)};

            actualExtent.width = std::clamp(actualExtent.width,
                                            capabilities.minImageExtent.width,
                                            capabilities.maxImageExtent.width);
            actualExtent.height =
                std::clamp(actualExtent.height,
                           capabilities.minImageExtent.height,
                           capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice device)
    {
        SwapchainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
            device, surface, &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(
            device, surface, &formatCount, nullptr);

        if (formatCount != 0)
        {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(
                device, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(
            device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0)
        {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(
                device,
                surface,
                &presentModeCount,
                details.presentModes.data());
        }

        return details;
    }

    bool isDeviceSuitable(VkPhysicalDevice device)
    {
        bool extensionsSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionsSupported)
        {
            SwapchainSupportDetails swapChainSupport =
                querySwapchainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() &&
                                !swapChainSupport.presentModes.empty();
        }

        VkPhysicalDeviceFeatures supportedFeatures;
        vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

        return extensionsSupported && swapChainAdequate &&
               supportedFeatures.samplerAnisotropy;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device)
    {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(
            device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(
            device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(),
                                                 deviceExtensions.end());

        for (const auto& extension : availableExtensions)
        {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    uint32_t tryFindGraphicsFamily()
    {
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(
            physicalDevice, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(
            physicalDevice, &queueFamilyCount, queueFamilies.data());

        int32_t i = 0;
        for (const auto& queueFamily : queueFamilies)
        {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                return i;
            }

            i++;
        }

        return -1;
    }

    std::vector<const char*> getRequiredExtensions()
    {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(
            glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers)
        {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    bool checkValidationLayerSupport()
    {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers)
        {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers)
            {
                if (strcmp(layerName, layerProperties.layerName) == 0)
                {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound)
            {
                return false;
            }
        }

        return true;
    }

    static std::vector<char> readFile(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        bool result = file.is_open();
        assert(result);

        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();

        return buffer;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL
        debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                      VkDebugUtilsMessageTypeFlagsEXT messageType,
                      const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                      void* pUserData)
    {
        std::cerr << pCallbackData->pMessage << std::endl;

        return VK_FALSE;
    }
};

int32_t main(void)
{
    Renderer app;
    app.run();
    return EXIT_SUCCESS;
}
