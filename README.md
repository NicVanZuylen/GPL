# GPL
> A GPGPU particle library using the Nvidia CUDA platform.

## About

GPL - GPU Particle Library, is a library designed to overcome standard video game particle system limitations by offloading
particle processing to the GPU. GPL uses Nvidia Corporation's CUDA platform to interface with the CUDA driver to allow GPGPU processing of
particles. GPL requires OpenGL to generate particle buffers and draw particlesystems efficiently. A Vulkan binding of GPL may come in the
near future.

## Features

* Create particlesystems using the GPL_ParticleSystem class and customize them using templated data structures.
* Create behaviour programs for particlesystems to define your own behaviour for particlesystems. (Similar to shader programs.)
* Shaders use the NVRTC library for easy runtime compilation of GPL behaviour programs.
* Use the CUDA GPGPU platform to process particles in parallel on the GPU.
* Specify your own vertex attributes for drawing GPL particles.
* Use your own shaders and meshes for drawing GPL particles, or draw the particles as pixels.

## Images

The following are some images taken from the demo project demonstrating some of the effects that can be acheived with GPL.

<a href="https://imgur.com/Qs95Iev"><img src="https://i.imgur.com/Qs95Iev.png" title="source: imgur.com" /></a>
<a href="https://imgur.com/3R0htE7"><img src="https://i.imgur.com/3R0htE7.png" title="source: imgur.com" /></a>
<a href="https://imgur.com/CLQ4WSN"><img src="https://i.imgur.com/CLQ4WSN.png" title="source: imgur.com" /></a>

## Libraries

* OpenGL
* GLFW3
* stb_image.h from https://github.com/nothings/stb
* TinyObjLoader - https://github.com/syoyo/tinyobjloader
* Nvidia CUDA Toolkit

## Clone

* Clone the GPL repository using this link: https://github.com/NicVanZuylen/GPL.git
* Or download the repository as a ZIP file.

## Building the project

The GPL demo project already has the correct configurations for building the library and demo application.

### Requirements:

* Nvidia CUDA Toolkit v10.1 or newer.

### Steps:

* Ensure the Nvidia CUDA Toolkit v10.1 or newer is installed.
* On Windows: Make sure the Windows SDK version is set to the version you have installed.
* Make sure the platform is set to x64.
* Build the GPL library: Go to GPL -> Build.
* Build the demo project: Go to GraphicsProject -> Build.
* The built project should now be possible to run!

## Demo Build

A demo build showcasing GPL is included in the repository.

### Demo controls

* W - Move forward.
* A - Move left.
* S - Move back.
* D - Move right.
* Right Click Hold - Look around.
* 1 - Run example particlesystem 1.
* 2 - Run example particlesystem 2.
* 3 - Run example particlesystem 3.
* Up Arrow - Move particle destination forward in worldspace.
* Down Arrow - Move particle destination back in worldspace.
* Left Arrow - Move particle destination left in worldspace.
* Right Arrow - Move particle destination right in worldspace.
* Page Up - Move particle destination up.
* Page Down - Move particle destination down.
* Escape - Exit application.
