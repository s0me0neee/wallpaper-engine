// Our replacement for Wallpaper Engine's own common.h.
//
// Prepended to every shader as well as being #include-able: several shaders
// call mul() and saturate() without including anything, so Wallpaper Engine
// must prepend it too. The guard makes the double safe.

#ifndef WE_COMMON_H
#define WE_COMMON_H

// M_PI_2 is two pi, not pi/2 — Wallpaper Engine's own spelling, and shaders
// rely on it for full-turn wrapping.
#define M_PI 3.14159265359
#define M_PI_HALF 1.57079632679
#define M_PI_2 6.28318530718
#define M_1_PI 0.31830988618

#define SQRT_2 1.41421356237
#define SQRT_3 1.73205080756

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

// HSV in the 0..1-per-component convention every shader here uses.
vec3 hsv2rgb(vec3 hsv)
{
	vec3 wrapped = abs(fract(hsv.xxx + vec3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
	return hsv.z * mix(vec3(1.0), clamp(wrapped - 1.0, 0.0, 1.0), hsv.y);
}

vec3 rgb2hsv(vec3 rgb)
{
	float high = max(max(rgb.r, rgb.g), rgb.b);
	float low = min(min(rgb.r, rgb.g), rgb.b);
	float chroma = high - low;

	float hue;
	if (chroma <= 0.0)      hue = 0.0;
	else if (rgb.r >= high) hue = (rgb.g - rgb.b) / chroma;
	else if (rgb.g >= high) hue = 2.0 + (rgb.b - rgb.r) / chroma;
	else                    hue = 4.0 + (rgb.r - rgb.g) / chroma;

	return vec3(fract(hue / 6.0), chroma / (high + 1e-10), high);
}

// The luma weights are Wallpaper Engine's own, and they are not the usual
// Rec.601 order — red is weighted 0.11 and blue 0.3. Matching it matters
// wherever a shader desaturates.
float greyscale(vec3 color)
{
	return dot(color, vec3(0.11, 0.59, 0.3));
}

#endif
