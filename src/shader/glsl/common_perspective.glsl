// The vertex-side quad transform.

#ifndef WE_COMMON_PERSPECTIVE_H
#define WE_COMMON_PERSPECTIVE_H

// Maps the unit square onto an arbitrary quad, as a homogeneous 3x3.
// Used by the perspective effect's vertex stage.
//
// A quad that is still a parallelogram — which is every one of these at rest —
// has no projective term, and the general solution divides by zero there. That
// case takes the affine map instead.
mat3 squareToQuad(vec2 p0, vec2 p1, vec2 p2, vec2 p3)
{
	vec2 d1 = p1 - p2;
	vec2 d2 = p3 - p2;
	vec2 sum = p0 - p1 + p2 - p3;
	float det = d1.x * d2.y - d2.x * d1.y;

	if (det == 0.0 || (sum.x == 0.0 && sum.y == 0.0))
	{
		return mat3(
			p1.x - p0.x, p1.y - p0.y, 0.0,
			p2.x - p1.x, p2.y - p1.y, 0.0,
			p0.x,        p0.y,        1.0);
	}

	float g = (sum.x * d2.y - d2.x * sum.y) / det;
	float h = (d1.x * sum.y - sum.x * d1.y) / det;

	return mat3(
		p1.x - p0.x + g * p1.x, p1.y - p0.y + g * p1.y, g,
		p3.x - p0.x + h * p3.x, p3.y - p0.y + h * p3.y, h,
		p0.x,                   p0.y,                   1.0);
}

#endif
