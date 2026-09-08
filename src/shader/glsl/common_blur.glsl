// Separable Gaussian taps.
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
// Weights are the standard linear-sampled Gaussian kernels for those tap
// counts, which is what the offsets' fractional spacing implies.

#ifndef WE_COMMON_BLUR_H
#define WE_COMMON_BLUR_H

vec4 weBlur3a(sampler2D image, vec2 uv, vec2 direction)
{
	vec4 color = texSample2D(image, uv) * 0.5;
	color += texSample2D(image, uv + direction) * 0.25;
	color += texSample2D(image, uv - direction) * 0.25;
	return color;
}

vec4 weBlur7a(sampler2D image, vec2 uv, vec2 direction)
{
	vec4 color = texSample2D(image, uv) * 0.3125;
	color += (texSample2D(image, uv + direction * 1.3333333333333333)
	        + texSample2D(image, uv - direction * 1.3333333333333333)) * 0.3125;
	color += (texSample2D(image, uv + direction * 3.111111111111111)
	        + texSample2D(image, uv - direction * 3.111111111111111)) * 0.03125;
	return color;
}

vec4 weBlur13a(sampler2D image, vec2 uv, vec2 direction)
{
	vec4 color = texSample2D(image, uv) * 0.1964825501511404;
	color += (texSample2D(image, uv + direction * 1.411764705882353)
	        + texSample2D(image, uv - direction * 1.411764705882353)) * 0.2969069646728344;
	color += (texSample2D(image, uv + direction * 3.2941176470588234)
	        + texSample2D(image, uv - direction * 3.2941176470588234)) * 0.09447039785044732;
	color += (texSample2D(image, uv + direction * 5.176470588235294)
	        + texSample2D(image, uv - direction * 5.176470588235294)) * 0.010381362401148057;
	return color;
}

#define blur3a(uv, direction) weBlur3a(g_Texture0, (uv), (direction))
#define blur7a(uv, direction) weBlur7a(g_Texture0, (uv), (direction))
#define blur13a(uv, direction) weBlur13a(g_Texture0, (uv), (direction))

#endif
