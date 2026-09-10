// Fragment-side helpers, and the texture-format table shaders branch on.
//
// The FORMAT_* numbers are not decoration: a shader asks
// `#if TEX2FORMAT == FORMAT_R8` to decide whether a sampler holds one channel
// or four. With the names undefined both sides of that comparison are 0, so it
// answers "yes" for every texture — `lightshafts.frag` then reads its gradient
// as `.rrr` and draws grey rays instead of coloured ones. The numbers are the
// enum Wallpaper Engine's own `.tex` header stores, so they have to match it.
//
// TEXnFORMAT itself is injected per bound texture by `shader::preprocess`,
// from the format in each `.tex` header. A sampler we cannot resolve leaves it
// undefined, which reads as FORMAT_RGBA8888 — the right guess for a texture
// that decoded to RGBA.

#ifndef WE_COMMON_FRAGMENT_H
#define WE_COMMON_FRAGMENT_H

#define FORMAT_RGBA8888 0
#define FORMAT_RGB888 1
#define FORMAT_RGB565 2
#define FORMAT_ETC1_RGB8 3
#define FORMAT_DXT5 4
#define FORMAT_ETC2_RGBA8 5
#define FORMAT_DXT3 6
#define FORMAT_DXT1 7
#define FORMAT_RG88 8
#define FORMAT_R8 9
#define FORMAT_RG1616F 10
#define FORMAT_R16F 11
#define FORMAT_BC7 12

// A slot whose format we could not resolve reads as RGBA — the right guess for
// anything that decoded to four channels, and the value that keeps a shader's
// `#if TEXnFORMAT == ...` a fully defined constant expression. `preprocess`
// emits its own #defines ahead of this, so a resolved slot wins.
#ifndef TEX0FORMAT
#define TEX0FORMAT FORMAT_RGBA8888
#endif
#ifndef TEX1FORMAT
#define TEX1FORMAT FORMAT_RGBA8888
#endif
#ifndef TEX2FORMAT
#define TEX2FORMAT FORMAT_RGBA8888
#endif
#ifndef TEX3FORMAT
#define TEX3FORMAT FORMAT_RGBA8888
#endif

// A single-channel sample lands in red on GL (in alpha on D3D9, which we never
// target).
float ConvertSampleR8(vec4 texel)
{
	return texel.r;
}

// One- and two-channel textures stand in for greyscale-plus-alpha, so they are
// expanded to the RGBA a shader expects to sample.
vec4 ConvertTextureFormat(const int format, vec4 texel)
{
	if (format == FORMAT_RG88 || format == FORMAT_RG1616F)
	{
		return texel.rrrg;
	}
	if (format == FORMAT_R8 || format == FORMAT_R16F)
	{
		return vec4(1.0, 1.0, 1.0, texel.r);
	}
	return texel;
}

vec4 ConvertTexture0Format(vec4 texel)
{
	return ConvertTextureFormat(TEX0FORMAT, texel);
}

// Whether slot 1 holds a block-compressed texture. Spelled as a chain of
// equalities rather than a range test because the GL compilers here reject a
// mixed `>=`/`&&`/`||` expression in `#if`.
#define WE_TEX1_BLOCK (TEX1FORMAT == FORMAT_ETC1_RGB8 || TEX1FORMAT == FORMAT_DXT5 || TEX1FORMAT == FORMAT_ETC2_RGBA8 || TEX1FORMAT == FORMAT_DXT3 || TEX1FORMAT == FORMAT_DXT1 || TEX1FORMAT == FORMAT_BC7)

// Normal maps are stored with the useful pair in whichever two channels the
// compression scheme keeps best, so unpacking depends on the slot's format.
// Z is reconstructed rather than stored.
vec3 DecompressNormal(vec4 normal)
{
#if WE_TEX1_BLOCK
	normal.yx = normal.yw * 2.0 - vec2(0.965, 1.0);
#elif TEX1FORMAT == FORMAT_RG88
	normal.xy = normal.rg * 2.0 - 1.0;
#else
	normal.xy = normal.wy * 2.0 - 1.0;
#endif
	normal.z = sqrt(saturate(1.0 - normal.x * normal.x - normal.y * normal.y));
	return normal.xyz;
}

// The same unpacking, keeping the fourth channel's mask intact.
vec4 DecompressNormalWithMask(vec4 normal)
{
#if WE_TEX1_BLOCK
	normal.xw = normal.wx;
	normal.xy = normal.xy * 2.0 - vec2(0.965, 1.0);
#elif TEX1FORMAT == FORMAT_RG88
	normal.xy = normal.gr * 2.0 - 1.0;
#else
	normal.xw = normal.wx;
	normal.xy = normal.xy * 2.0 - 1.0;
#endif
	normal.z = sqrt(saturate(1.0 - normal.x * normal.x - normal.y * normal.y));
	return normal;
}

// Deliberately absent: `ComputeLight`, `ComputeLightSpecular` and the
// `ComputeMaterialSpecular*` pair. Only Wallpaper Engine's own model shaders
// call them, and no wallpaper packages those — a wallpaper ships effect
// shaders, which is why every scene in the corpus includes this header for the
// FORMAT_* table above and nothing else. A lighting model nothing calls is
// guesswork we could not check against anything; a shader that needs one
// should fail loudly here rather than render subtly wrong.

#endif
