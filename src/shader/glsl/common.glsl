// Our replacement for Wallpaper Engine's own common.h.
//
// Prepended to every shader as well as being #include-able: several shaders
// call mul() and saturate() without including anything, so Wallpaper Engine
// must prepend it too. The guard makes the double safe.

#ifndef WE_COMMON_H
#define WE_COMMON_H

#define M_PI 3.14159265359
#define M_PI_2 6.28318530718
#define M_1_PI 0.31830988618

// HLSL splat constructors. CAST4(x) is (float4)x, i.e. x in every lane.
#define CAST2(x) vec2(x)
#define CAST3(x) vec3(x)
#define CAST4(x) vec4(x)
#define CAST3X3(x) mat3(x)

// HLSL's mul(v, M) is the row-vector convention: the vector goes on the left.
// GLSL's equivalent spelling is M * v, so the operands swap. Every call in the
// corpus is vector-by-matrix; a matrix-by-matrix call would need the operands
// left alone, so this must be revisited if one ever appears.
#define mul(a, b) ((b) * (a))

#define saturate(x) clamp((x), 0.0, 1.0)
#define frac(x) fract(x)
#define lerp(a, b, t) mix((a), (b), (t))
#define rsqrt(x) inversesqrt(x)
#define atan2(y, x) atan((y), (x))
#define fmod(a, b) mod((a), (b))
#define ddx(x) dFdx(x)
#define ddy(x) dFdy(x)

#define texSample2D(s, uv) texture((s), (uv))
#define texSample2DLod(s, uv, lod) textureLod((s), (uv), (lod))

// Rotate a 2D vector counter-clockwise by `angle` radians.
vec2 rotateVec2(vec2 v, float angle)
{
	float s = sin(angle);
	float c = cos(angle);
	return vec2(v.x * c - v.y * s, v.x * s + v.y * c);
}

#endif
