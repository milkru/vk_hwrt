#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable

struct Vertex
{
	float pos[3];
	float norm[3];
};

layout(location = 0) rayPayloadInEXT vec3 payload;
layout(location = 1) rayPayloadEXT bool inShadow;
hitAttributeEXT vec2 hitAttrib;

layout(binding = 0, set = 0) uniform accelerationStructureEXT tlas;
layout(binding = 2, set = 0) uniform Ubo 
{
	mat4 viewInverse;
	mat4 projInverse;
	vec4 lightDir;
} ubo;
layout(binding = 3, set = 0) readonly buffer Vertices { Vertex vertices[]; };
layout(binding = 4, set = 0) readonly buffer Indices { uint indices[]; };

void main()
{
	uint i0 = indices[3 * gl_PrimitiveID + 0];
	uint i1 = indices[3 * gl_PrimitiveID + 1];
	uint i2 = indices[3 * gl_PrimitiveID + 2];

	Vertex v0 = vertices[i0];
	Vertex v1 = vertices[i1];
	Vertex v2 = vertices[i2];

	vec3 n0 = vec3(v0.norm[0], v0.norm[1], v0.norm[2]);
	vec3 n1 = vec3(v1.norm[0], v1.norm[1], v1.norm[2]);
	vec3 n2 = vec3(v2.norm[0], v2.norm[1], v2.norm[2]);

    vec3 bcCoords = vec3(1.0f - hitAttrib.x - hitAttrib.y, hitAttrib.x, hitAttrib.y);
	vec3 normal = normalize(n0 * bcCoords.x + n1 * bcCoords.y + n2 * bcCoords.z);

    float diffuseFactor = max(dot(-ubo.lightDir.xyz, normal), 0.0);

	inShadow = true;
    traceRayEXT(
		tlas,
		gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT,
		0xff,
		/*sbt_offset, second miss shader is offseted by 1 in sbt*/ 1, 0, /*miss shader index*/ 1,
		gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT,
		0.001,
		-ubo.lightDir.xyz,
		10000.0,
		/*inShadow payload location index*/ 1);
	diffuseFactor = 1;
	if (inShadow)
	{
		diffuseFactor = 0.3 * diffuseFactor;
	}

	vec3 ambientColor = vec3(0.28125, 0.38672, 0.60937);
    vec3 ambientComponent = 0.2 * ambientColor;

	vec3 diffuseColor = vec3(0.96484, 0.597656, 0.42969);
    vec3 diffuseComponent = clamp(diffuseColor * diffuseFactor, 0, 1);

    payload = ambientComponent + diffuseComponent;
}
