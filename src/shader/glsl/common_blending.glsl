// Blend-mode dispatch.
//
// The mode arrives as an argument, not a preprocessor value: every call site
// in the corpus is ApplyBlending(BLENDMODE, base, blend, alpha), so this is a
// run-time branch on a constant, which the driver folds away.
//
// Mode numbering follows Wallpaper Engine's own blend-mode dropdown order.

#ifndef WE_COMMON_BLENDING_H
#define WE_COMMON_BLENDING_H

vec3 BlendColor(int mode, vec3 base, vec3 blend)
{
	if (mode == 1) return base + blend;                          // add
	if (mode == 2) return base * blend;                          // multiply
	if (mode == 3) return 1.0 - (1.0 - base) * (1.0 - blend);    // screen
	if (mode == 4) return base - blend;                          // subtract
	if (mode == 5) return abs(base - blend);                     // difference
	if (mode == 6) return min(base, blend);                      // darken
	if (mode == 7) return max(base, blend);                      // lighten
	return blend;                                                // normal
}

// `alpha` is the blend layer's coverage: 0 leaves the base untouched.
vec3 ApplyBlending(int mode, vec3 base, vec3 blend, float alpha)
{
	return mix(base, BlendColor(mode, base, blend), alpha);
}

#endif
