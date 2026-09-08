// The vertex-side quad transform.

#ifndef WE_COMMON_PERSPECTIVE_H
#define WE_COMMON_PERSPECTIVE_H

// Maps the unit square onto an arbitrary quad, as a homogeneous 3x3.
// Used by the perspective effect's vertex stage.
mat3 squareToQuad(vec2 p0, vec2 p1, vec2 p2, vec2 p3)
{
	float dx1 = p1.x - p2.x;
	float dy1 = p1.y - p2.y;
	float dx2 = p3.x - p2.x;
	float dy2 = p3.y - p2.y;
	float sx = p0.x - p1.x + p2.x - p3.x;
	float sy = p0.y - p1.y + p2.y - p3.y;
	float den = dx1 * dy2 - dx2 * dy1;

	float g = (sx * dy2 - dx2 * sy) / den;
	float h = (dx1 * sy - sx * dy1) / den;

	return mat3(
		p1.x - p0.x + g * p1.x, p1.y - p0.y + g * p1.y, g,
		p3.x - p0.x + h * p3.x, p3.y - p0.y + h * p3.y, h,
		p0.x,                   p0.y,                   1.0);
}

#endif
