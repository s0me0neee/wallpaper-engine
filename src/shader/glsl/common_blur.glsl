// Separable blur taps.
//
// Every call site is blurNa(uv, direction) with no sampler argument, so the
// source texture is fixed: these only ever run in a gaussian pass, where the
// image being blurred is g_Texture0.
//
// The sampler is bound by a macro rather than named inside the functions,
// because one shader includes this header *above* its own g_Texture0
// declaration. A function body referring to it there would not compile;
// a macro expands at the call site, where the uniform is always in scope.
//
// The offsets and weights are the kernels Wallpaper Engine's own passes use,
// and have to match tap for tap: they set how far a blur reaches, so an
// invented kernel of the same tap count still blurs by the wrong amount.
// blur7a in particular is not symmetric — it is four taps at +2.352, +0.469,
// -1.409 and -3.0, which an earlier symmetric five-tap guess here got wrong.

#ifndef WE_COMMON_BLUR_H
#define WE_COMMON_BLUR_H

vec4 weBlur3a(sampler2D image, vec2 uv, vec2 direction)
{
	return texSample2D(image, uv + direction) * 0.25
	     + texSample2D(image, uv) * 0.5
	     + texSample2D(image, uv - direction) * 0.25;
}

vec4 weBlur7a(sampler2D image, vec2 uv, vec2 direction)
{
	vec2 o1 = 2.3515644035337887 * direction;
	vec2 o2 = 0.469433779698372 * direction;
	vec2 o3 = 1.4091998770852121 * direction;
	vec2 o4 = 3.0 * direction;
	return texSample2D(image, uv + o1) * 0.2028175528299753
	     + texSample2D(image, uv + o2) * 0.4044856614512112
	     + texSample2D(image, uv - o3) * 0.3213933537319605
	     + texSample2D(image, uv - o4) * 0.0713034319868530;
}

vec4 weBlur13a(sampler2D image, vec2 uv, vec2 direction)
{
	vec2 o1 = 1.4091998770852122 * direction;
	vec2 o2 = 3.2979348079914822 * direction;
	vec2 o3 = 5.2062900776825969 * direction;
	return texSample2D(image, uv) * 0.1976406528809576
	     + (texSample2D(image, uv + o1) + texSample2D(image, uv - o1)) * 0.2959855056006557
	     + (texSample2D(image, uv + o2) + texSample2D(image, uv - o2)) * 0.0935333619980593
	     + (texSample2D(image, uv + o3) + texSample2D(image, uv - o3)) * 0.0116608059608062;
}

// The radial variants sweep the same taps around a centre instead of along a
// direction, by the angle each offset implies. `amount` is in the effect's own
// units, a fortieth of a radian per unit.
vec2 weBlurRotate(vec2 v, float angle)
{
	float s = sin(angle);
	float c = cos(angle);
	return vec2(v.x * c - v.y * s, v.x * s + v.y * c);
}

// The name this header exports it under; identical to common.h's rotateVec2,
// which is why the radial blurs above use it rather than redefining the maths.
vec2 blurRotateVec2(vec2 v, float angle)
{
	return weBlurRotate(v, angle);
}

vec4 weBlurRadial3a(sampler2D image, vec2 uv, vec2 center, float amount)
{
	vec2 delta = uv - center;
	vec2 r1 = weBlurRotate(delta, amount * 0.025) - delta;
	return texSample2D(image, uv) * 0.5
	     + texSample2D(image, uv + r1) * 0.25
	     + texSample2D(image, uv - r1) * 0.25;
}

vec4 weBlurRadial7a(sampler2D image, vec2 uv, vec2 center, float amount)
{
	vec2 delta = uv - center;
	float a = amount * 0.025;
	vec2 r1 = weBlurRotate(delta, 2.3515644035337887 * a) - delta;
	vec2 r2 = weBlurRotate(delta, 0.469433779698372 * a) - delta;
	vec2 r3 = weBlurRotate(delta, -1.4091998770852121 * a) - delta;
	vec2 r4 = weBlurRotate(delta, -3.0 * a) - delta;
	return texSample2D(image, uv + r1) * 0.2028175528299753
	     + texSample2D(image, uv + r2) * 0.4044856614512112
	     + texSample2D(image, uv + r3) * 0.3213933537319605
	     + texSample2D(image, uv + r4) * 0.0713034319868530;
}

vec4 weBlurRadial13a(sampler2D image, vec2 uv, vec2 center, float amount)
{
	vec2 delta = uv - center;
	float a = amount * 0.025;
	vec2 r1 = weBlurRotate(delta, 1.4091998770852122 * a) - delta;
	vec2 r2 = weBlurRotate(delta, 3.2979348079914822 * a) - delta;
	vec2 r3 = weBlurRotate(delta, 5.2062900776825969 * a) - delta;
	return texSample2D(image, uv) * 0.1976406528809576
	     + (texSample2D(image, uv + r1) + texSample2D(image, uv - r1)) * 0.2959855056006557
	     + (texSample2D(image, uv + r2) + texSample2D(image, uv - r2)) * 0.0935333619980593
	     + (texSample2D(image, uv + r3) + texSample2D(image, uv - r3)) * 0.0116608059608062;
}

#define blur3a(uv, direction) weBlur3a(g_Texture0, (uv), (direction))
#define blur7a(uv, direction) weBlur7a(g_Texture0, (uv), (direction))
#define blur13a(uv, direction) weBlur13a(g_Texture0, (uv), (direction))
#define blur3(uv, direction) (weBlur3a(g_Texture0, (uv), (direction)).rgb)
#define blur7(uv, direction) (weBlur7a(g_Texture0, (uv), (direction)).rgb)
#define blur13(uv, direction) (weBlur13a(g_Texture0, (uv), (direction)).rgb)
#define blurRadial3a(uv, center, amount) weBlurRadial3a(g_Texture0, (uv), (center), (amount))
#define blurRadial7a(uv, center, amount) weBlurRadial7a(g_Texture0, (uv), (center), (amount))
#define blurRadial13a(uv, center, amount) weBlurRadial13a(g_Texture0, (uv), (center), (amount))

#endif
