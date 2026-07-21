#version 460
#extension GL_EXT_scalar_block_layout : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _12
{
    float _m0;
    float _m1;
    float _m2;
    float _m3;
};

struct _13
{
    mat4 _m0;
    mat3 _m1;
    _12 _m2;
    _12 _m3;
    mat3x2 _m4;
    mat3x2 _m5;
    uvec2 _m6;
    uint _m7;
};

layout(set = 0, binding = 3, std430) uniform _20_19
{
    _13 _m0;
} _19;

layout(set = 0, binding = 0) uniform texture2D _15;
layout(set = 0, binding = 1) uniform texture2D _17;
layout(set = 0, binding = 2) uniform texture2D _18;

uint _30(texture2D _23, texture2D _25, texture2D _27, _13 _29)
{
    return 0u;
}

uint _42(texture2D _35, texture2D _37, texture2D _39, _13 _41)
{
    return _30(_35, _37, _39, _41);
}

void main()
{
}

