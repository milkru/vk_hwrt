#version 460
#extension GL_EXT_ray_tracing : enable

layout(location = 0) rayPayloadInEXT vec3 payload;

void main()
{
    payload = vec3(0.28125, 0.38672, 0.60937);
}
