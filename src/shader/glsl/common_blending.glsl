// Blend-mode dispatch.
//
// The mode arrives as an argument, not a preprocessor value: every call site
// in the corpus is ApplyBlending(BLENDMODE, base, blend, alpha), so this is a
// run-time branch on a constant the driver folds away. Wallpaper Engine's own
// header instead compiles one `#if BLENDMODE == n` arm; same result, and one
// function is easier to read than thirty-two conditional returns.
//
// The numbering is not a preference — it is what `scene.json` stores, so it
// has to match: 9 is the value `shine_combine` and `godrays_combine` default
// to, 31 is `lightshafts`'s, 32 is `caustics`'s. An earlier eight-mode guess
// here made every mode above 7 fall through to "normal", which turned those
// combine passes into a plain overwrite of the layer.
//
// Modes 5 and 10 ignore `alpha`, and 31 and 32 fold it in themselves, so the
// mix is per-mode rather than one at the end. The formulas are the usual
// compositing set (Photoshop's, as generalised by PEGTOP), and the four HSL
// modes go through a standard RGB<->HSL round trip.

#ifndef WE_COMMON_BLENDING_H
#define WE_COMMON_BLENDING_H

vec3 weRgbToHsl(vec3 color)
{
	float low = min(min(color.r, color.g), color.b);
	float high = max(max(color.r, color.g), color.b);
	float delta = high - low;
	float lightness = (high + low) * 0.5;

	if (delta <= 0.0)
	{
		return vec3(0.0, 0.0, lightness);
	}

	float saturation = lightness < 0.5 ? delta / (high + low) : delta / (2.0 - high - low);

	// Hue as sixths of the wheel, from whichever channel is the maximum.
	vec3 d = (vec3(high) - color) / (6.0 * delta) + 0.5;
	float hue;
	if (color.r >= high)      hue = d.b - d.g;
	else if (color.g >= high) hue = (1.0 / 3.0) + d.r - d.b;
	else                      hue = (2.0 / 3.0) + d.g - d.r;

	return vec3(fract(hue), saturation, lightness);
}

float weHueToChannel(float f1, float f2, float hue)
{
	hue = fract(hue);
	if (6.0 * hue < 1.0) return f1 + (f2 - f1) * 6.0 * hue;
	if (2.0 * hue < 1.0) return f2;
	if (3.0 * hue < 2.0) return f1 + (f2 - f1) * ((2.0 / 3.0) - hue) * 6.0;
	return f1;
}

vec3 weHslToRgb(vec3 hsl)
{
	if (hsl.y <= 0.0)
	{
		return vec3(hsl.z);
	}

	float f2 = hsl.z < 0.5 ? hsl.z * (1.0 + hsl.y) : (hsl.z + hsl.y) - (hsl.y * hsl.z);
	float f1 = 2.0 * hsl.z - f2;
	return vec3(
		weHueToChannel(f1, f2, hsl.x + 1.0 / 3.0),
		weHueToChannel(f1, f2, hsl.x),
		weHueToChannel(f1, f2, hsl.x - 1.0 / 3.0));
}

// The conversions and the individual blend modes are part of this header's
// public surface, not just plumbing for ApplyBlending: Workshop effect shaders
// call them directly — `scene_example6`'s colour grading uses RGBToHSL and
// HSLToRGB, `scene_example8`'s uses BlendSoftLight.
vec3 RGBToHSL(vec3 color)
{
	return weRgbToHsl(color);
}

vec3 HSLToRGB(vec3 hsl)
{
	return weHslToRgb(hsl);
}

float HueToRGB(float f1, float f2, float hue)
{
	return weHueToChannel(f1, f2, hue);
}

vec4 Desaturate(vec3 color, float desaturation)
{
	vec3 grey = vec3(dot(vec3(0.3, 0.59, 0.11), color));
	return vec4(mix(color, grey, desaturation), 1.0);
}

vec3 ContrastSaturationBrightness(vec3 color, float brightness, float saturation, float contrast)
{
	const vec3 luma = vec3(0.2125, 0.7154, 0.0721);
	vec3 bright = color * brightness;
	vec3 saturated = mix(vec3(dot(bright, luma)), bright, saturation);
	return mix(vec3(0.5), saturated, contrast);
}

// Guarded so a zero (or one) divisor lands on the endpoint the mode defines
// rather than on an infinity.
vec3 weColorBurn(vec3 base, vec3 blend)
{
	return max(1.0 - (1.0 - base) / max(blend, vec3(1e-6)), vec3(0.0));
}

vec3 weColorDodge(vec3 base, vec3 blend)
{
	vec3 dodged = min(base / max(1.0 - blend, vec3(1e-6)), vec3(1.0));
	return mix(dodged, vec3(1.0), step(vec3(1.0), blend));
}

vec3 weReflect(vec3 base, vec3 blend)
{
	vec3 reflected = min(base * base / max(1.0 - blend, vec3(1e-6)), vec3(1.0));
	return mix(reflected, vec3(1.0), step(vec3(1.0), blend));
}

vec3 weLinearBurn(vec3 base, vec3 blend)
{
	return max(base + blend - 1.0, vec3(0.0));
}

vec3 weOverlay(vec3 base, vec3 blend)
{
	vec3 low = 2.0 * base * blend;
	vec3 high = 1.0 - 2.0 * (1.0 - base) * (1.0 - blend);
	return mix(low, high, step(vec3(0.5), base));
}

vec3 weSoftLight(vec3 base, vec3 blend)
{
	vec3 low = 2.0 * base * blend + base * base * (1.0 - 2.0 * blend);
	vec3 high = sqrt(base) * (2.0 * blend - 1.0) + 2.0 * base * (1.0 - blend);
	return mix(low, high, step(vec3(0.5), blend));
}

vec3 weVividLight(vec3 base, vec3 blend)
{
	vec3 low = weColorBurn(base, 2.0 * blend);
	vec3 high = weColorDodge(base, 2.0 * (blend - 0.5));
	return mix(low, high, step(vec3(0.5), blend));
}

vec3 weLinearLight(vec3 base, vec3 blend)
{
	vec3 low = weLinearBurn(base, 2.0 * blend);
	vec3 high = base + 2.0 * (blend - 0.5);
	return mix(low, high, step(vec3(0.5), blend));
}

vec3 wePinLight(vec3 base, vec3 blend)
{
	vec3 low = min(base, 2.0 * blend);
	vec3 high = max(base, 2.0 * (blend - 0.5));
	return mix(low, high, step(vec3(0.5), blend));
}

vec3 weBlendColor(int mode, vec3 base, vec3 blend)
{
	if (mode == 1)  return min(base, blend);                              // darken
	if (mode == 2)  return base * blend;                                  // multiply
	if (mode == 3)  return weColorBurn(base, blend);
	if (mode == 4)  return weLinearBurn(base, blend);                     // "substract"
	if (mode == 6)  return max(base, blend);                              // lighten
	if (mode == 7)  return 1.0 - (1.0 - base) * (1.0 - blend);            // screen
	if (mode == 8)  return weColorDodge(base, blend);
	if (mode == 9)  return min(base + blend, vec3(1.0));                  // add
	if (mode == 11) return weOverlay(base, blend);
	if (mode == 12) return weSoftLight(base, blend);
	if (mode == 13) return weOverlay(blend, base);                        // hard light
	if (mode == 14) return weVividLight(base, blend);
	if (mode == 15) return weLinearLight(base, blend);
	if (mode == 16) return wePinLight(base, blend);
	if (mode == 17) return step(vec3(0.5), weVividLight(base, blend));    // hard mix
	if (mode == 18) return abs(base - blend);                             // difference
	if (mode == 19) return base + blend - 2.0 * base * blend;             // exclusion
	if (mode == 20) return weLinearBurn(base, blend);                     // subtract again
	if (mode == 21) return weReflect(base, blend);
	if (mode == 22) return weReflect(blend, base);                        // glow
	if (mode == 23) return min(base, blend) - max(base, blend) + 1.0;     // phoenix
	if (mode == 24) return (base + blend) * 0.5;                          // average
	if (mode == 25) return 1.0 - abs(1.0 - base - blend);                 // negation
	if (mode == 26) return weHslToRgb(vec3(weRgbToHsl(blend).x, weRgbToHsl(base).yz));
	if (mode == 27) return weHslToRgb(vec3(weRgbToHsl(base).x, weRgbToHsl(blend).y, weRgbToHsl(base).z));
	if (mode == 28) return weHslToRgb(vec3(weRgbToHsl(blend).xy, weRgbToHsl(base).z));
	if (mode == 29) return weHslToRgb(vec3(weRgbToHsl(base).xy, weRgbToHsl(blend).z));
	if (mode == 30) return max(base.x, max(base.y, base.z)) * blend;      // tint
	return blend;                                                         // normal
}

// The single-channel forms. Wallpaper Engine's header builds its vec3 modes
// out of these, and a shader that wants one channel calls them directly, so
// they are part of the surface even though nothing here needs them.
#define BlendLinearDodgef(base, blend) ((base) + (blend))
#define BlendLinearBurnf(base, blend) max((base) + (blend) - 1.0, 0.0)
#define BlendLightenf(base, blend) max((blend), (base))
#define BlendDarkenf(base, blend) min((blend), (base))
#define BlendScreenf(base, blend) (1.0 - ((1.0 - (base)) * (1.0 - (blend))))
#define BlendOverlayf(base, blend) ((base) < 0.5 ? (2.0 * (base) * (blend)) : (1.0 - 2.0 * (1.0 - (base)) * (1.0 - (blend))))
#define BlendSoftLightf(base, blend) ((blend) < 0.5 ? (2.0 * (base) * (blend) + (base) * (base) * (1.0 - 2.0 * (blend))) : (sqrt(base) * (2.0 * (blend) - 1.0) + 2.0 * (base) * (1.0 - (blend))))
#define BlendColorDodgef(base, blend) ((blend) == 1.0 ? (blend) : min((base) / (1.0 - (blend)), 1.0))
#define BlendColorBurnf(base, blend) ((blend) == 0.0 ? (blend) : max(1.0 - ((1.0 - (base)) / (blend)), 0.0))
#define BlendLinearLightf(base, blend) ((blend) < 0.5 ? BlendLinearBurnf((base), (2.0 * (blend))) : BlendLinearDodgef((base), (2.0 * ((blend) - 0.5))))
#define BlendVividLightf(base, blend) ((blend) < 0.5 ? BlendColorBurnf((base), (2.0 * (blend))) : BlendColorDodgef((base), (2.0 * ((blend) - 0.5))))
#define BlendPinLightf(base, blend) ((blend) < 0.5 ? BlendDarkenf((base), (2.0 * (blend))) : BlendLightenf((base), (2.0 * ((blend) - 0.5))))
#define BlendHardMixf(base, blend) (BlendVividLightf((base), (blend)) < 0.5 ? 0.0 : 1.0)
#define BlendReflectf(base, blend) ((blend) == 1.0 ? (blend) : min((base) * (base) / (1.0 - (blend)), 1.0))

// Each mode under the name a shader calls it by. Macros rather than functions
// so the pair-swapping ones (hard light, glow) cost nothing.
#define BlendNormal(base, blend) (blend)
#define BlendDarken(base, blend) min((base), (blend))
#define BlendLighten(base, blend) max((base), (blend))
#define BlendMultiply(base, blend) ((base) * (blend))
#define BlendAverage(base, blend) (((base) + (blend)) * 0.5)
#define BlendAdd(base, blend) min((base) + (blend), vec3(1.0))
#define BlendSubstract(base, blend) weLinearBurn((base), (blend))
#define BlendLinearBurn(base, blend) weLinearBurn((base), (blend))
#define BlendLinearDodge(base, blend) min((base) + (blend), vec3(1.0))
#define BlendDifference(base, blend) abs((base) - (blend))
#define BlendNegation(base, blend) (1.0 - abs(1.0 - (base) - (blend)))
#define BlendExclusion(base, blend) ((base) + (blend) - 2.0 * (base) * (blend))
#define BlendScreen(base, blend) (1.0 - (1.0 - (base)) * (1.0 - (blend)))
#define BlendOverlay(base, blend) weOverlay((base), (blend))
#define BlendHardLight(base, blend) weOverlay((blend), (base))
#define BlendSoftLight(base, blend) weSoftLight((base), (blend))
#define BlendColorDodge(base, blend) weColorDodge((base), (blend))
#define BlendColorBurn(base, blend) weColorBurn((base), (blend))
#define BlendLinearLight(base, blend) weLinearLight((base), (blend))
#define BlendVividLight(base, blend) weVividLight((base), (blend))
#define BlendPinLight(base, blend) wePinLight((base), (blend))
#define BlendHardMix(base, blend) step(vec3(0.5), weVividLight((base), (blend)))
#define BlendReflect(base, blend) weReflect((base), (blend))
#define BlendGlow(base, blend) weReflect((blend), (base))
#define BlendPhoenix(base, blend) (min((base), (blend)) - max((base), (blend)) + 1.0)
#define BlendTint(base, blend) (max((base).x, max((base).y, (base).z)) * (blend))
#define BlendOpacity(base, blend, mode, opacity) mix((base), mode((base), (blend)), (opacity))

vec3 BlendHue(vec3 base, vec3 blend)
{
	return weBlendColor(26, base, blend);
}

vec3 BlendSaturation(vec3 base, vec3 blend)
{
	return weBlendColor(27, base, blend);
}

vec3 BlendColor(vec3 base, vec3 blend)
{
	return weBlendColor(28, base, blend);
}

vec3 BlendLuminosity(vec3 base, vec3 blend)
{
	return weBlendColor(29, base, blend);
}

// `alpha` is the blend layer's coverage: 0 leaves the base untouched.
vec3 ApplyBlending(int mode, vec3 base, vec3 blend, float alpha)
{
	if (mode == 5)  return min(base, blend);
	if (mode == 10) return max(base, blend);
	if (mode == 31) return base + blend * alpha;
	if (mode == 32) return mix(base, base + base * blend, alpha);
	return mix(base, weBlendColor(mode, base, blend), alpha);
}

#endif
