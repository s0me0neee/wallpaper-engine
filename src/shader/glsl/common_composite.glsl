// The "composite" controls newer stock effects (blur_combine) share: where the
// processed image lands relative to the layer it came from. COMPOSITE picks the
// mode (0 replace, 1 blend over, 2 under, 3 only where the layer is empty),
// and the three uniforms below are ordinary material values with annotated
// defaults, which is why the shim parses headers for annotations too.

#ifndef WE_COMMON_COMPOSITE_H
#define WE_COMMON_COMPOSITE_H

#include "common.h"
#include "common_blending.h"

uniform float g_CompositeAlpha; // {"material":"compositealpha","default":1}
uniform vec2 g_CompositeOffset; // {"material":"compositeoffset","default":"0 0"}
uniform vec3 g_CompositeColor; // {"material":"compositecolor","default":"1 1 1","type":"color"}

// Shift the sample point by the offset in texels, when compositing at all.
vec2 ApplyCompositeOffset(vec2 texCoords, vec2 textureResolution)
{
#if COMPOSITE
	texCoords += g_CompositeOffset / textureResolution;
#endif
	return texCoords;
}

vec4 ApplyComposite(vec4 original, vec4 effect)
{
#if COMPOSITEMONO == 1
	effect.rgb = vec3(greyscale(effect.rgb));
#endif
	effect.rgb *= g_CompositeColor;

#if COMPOSITE == 1
	effect.rgb = ApplyBlending(BLENDMODE, original.rgb, effect.rgb, effect.a * g_CompositeAlpha);
	effect.a = max(effect.a * saturate(g_CompositeAlpha), original.a);
#elif COMPOSITE == 2
	effect.a *= saturate(g_CompositeAlpha);
	effect = mix(effect, original, original.a);
#elif COMPOSITE == 3
	effect.a *= saturate(g_CompositeAlpha) * (1.0 - original.a);
#endif
	return effect;
}

#endif
