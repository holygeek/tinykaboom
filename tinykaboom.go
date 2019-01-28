package main

import (
	"bytes"
	"fmt"
	"log"
	"math"
	"os"
	"runtime"
	"sync"
)

type Vec struct {
	x, y, z float64
}

func NewVec(x, y, z float64) *Vec {
	return &Vec{
		x: x,
		y: y,
		z: z,
	}
}

func (v *Vec) Dot(o *Vec) float64 {
	return v.x*o.x + v.y*o.y + v.z*o.z
}

func (v *Vec) Mul(n float64) *Vec {
	return &Vec{
		x: v.x * n,
		y: v.y * n,
		z: v.z * n,
	}
}

func (v *Vec) Add(o *Vec) *Vec {
	return &Vec{
		x: v.x + o.x,
		y: v.y + o.y,
		z: v.z + o.z,
	}
}

func (v *Vec) Sub(o *Vec) *Vec {
	return &Vec{
		x: v.x - o.x,
		y: v.y - o.y,
		z: v.z - o.z,
	}
}

func (v *Vec) Norm() float64 {
	return math.Sqrt(v.x*v.x + v.y*v.y + v.z*v.z)
}

func (v *Vec) Normalize(l float64) *Vec {
	d := l / v.Norm()
	v.x = v.x * d
	v.y = v.y * d
	v.z = v.z * d
	return v
}

const (
	sphere_radius   = 1.5 // all the explosion fits in a sphere with this radius. The center lies in the origin.
	noise_amplitude = 1.0 // amount of noise applied to the sphere (towards the center)
)

func lerpFloat64(v0, v1, t float64) float64 {
	return v0 + (v1-v0)*math.Max(0.0, math.Min(1.0, t))
}

func lerpVec(v0, v1 *Vec, t float64) *Vec {
	return v0.Add((v1.Sub(v0)).Mul(math.Max(0.0, math.Min(1.0, t))))
}

func hash(n float64) float64 {
	x := math.Sin(n) * 43758.5453
	return x - math.Floor(x)
}

func noise(x *Vec) float64 {
	p := &Vec{x: math.Floor(x.x), y: math.Floor(x.y), z: math.Floor(x.z)}
	f := &Vec{x: x.x - p.x, y: x.y - p.y, z: x.z - p.z}
	f = f.Mul(f.Dot(NewVec(3, 3, 3).Sub(f.Mul(2))))
	n := p.Dot(NewVec(1, 57, 113))

	return lerpFloat64(lerpFloat64(
		lerpFloat64(hash(n+0), hash(n+1), f.x),
		lerpFloat64(hash(n+57), hash(n+58), f.x), f.y),
		lerpFloat64(
			lerpFloat64(hash(n+113), hash(n+114), f.x),
			lerpFloat64(hash(n+170), hash(n+171), f.x), f.y), f.z)
}

func rotate(v *Vec) *Vec {
	return NewVec(NewVec(0.00, 0.80, 0.60).Dot(v), NewVec(-0.80, 0.36, -0.48).Dot(v), NewVec(-0.60, -0.48, 0.64).Dot(v))
}

func fractal_brownian_motion(x *Vec) float64 { // this is a bad noise function with lots of artifacts. TODO: find a better one
	p := rotate(x)
	f := 0.0
	f += 0.5000 * noise(p)
	p = p.Mul(2.32)
	f += 0.2500 * noise(p)
	p = p.Mul(3.03)
	f += 0.1250 * noise(p)
	p = p.Mul(2.61)
	f += 0.0625 * noise(p)
	return f / 0.9375
}

func palette_fire(d float64) *Vec { // simple linear gradent yellow-orange-red-darkgray-gray. d is supposed to vary from 0 to 1
	var (
		yellow   = NewVec(1.7, 1.3, 1.0) // note that the color is "hot", i.e. has components >1
		orange   = NewVec(1.0, 0.6, 0.0)
		red      = NewVec(1.0, 0.0, 0.0)
		darkgray = NewVec(0.2, 0.2, 0.2)
		gray     = NewVec(0.4, 0.4, 0.4)
	)

	x := math.Max(0, math.Min(1, d))
	if x < .25 {
		return lerpVec(gray, darkgray, x*4)
	} else if x < .5 {
		return lerpVec(darkgray, red, x*4-1)
	} else if x < .75 {
		return lerpVec(red, orange, x*4-2)
	}
	return lerpVec(orange, yellow, x*4-3)
}

func signed_distance(p *Vec) float64 { // this function defines the implicit surface we render
	displacement := -fractal_brownian_motion(p.Mul(3.4)) * noise_amplitude
	return p.Norm() - (sphere_radius + displacement)
}

func sphere_trace(orig, dir, pos *Vec) bool { // Notice the early discard; in fact I know that the noise() function produces non-negative values,
	if orig.Dot(orig)-math.Pow(orig.Dot(dir), 2) > math.Pow(sphere_radius, 2) {
		return false // thus all the explosion fits in the sphere. Thus this early discard is a conservative check.
	}
	// It is not necessary, just a small speed-up
	*pos = *orig
	for i := 0; i < 128; i++ {
		d := signed_distance(pos)
		if d < 0 {
			return true
		}
		*pos = *(pos.Add(dir.Mul(math.Max(d*0.1, .01)))) // note that the step depends on the current distance, if we are far from the surface, we can do big steps
	}
	return false
}

func distance_field_normal(pos *Vec) *Vec { // simple finite differences, very sensitive to the choice of the eps constant
	const eps = 0.1
	d := signed_distance(pos)
	nx := signed_distance(NewVec(eps, 0, 0).Add(pos)) - d
	ny := signed_distance(NewVec(0, eps, 0).Add(pos)) - d
	nz := signed_distance(NewVec(0, 0, eps).Add(pos)) - d
	return NewVec(nx, ny, nz).Normalize(1)
}

func main() {
	const (
		width  = 640         // image width
		height = 480         // image height
		fov    = math.Pi / 3 // field of view angle
	)

	framebuffer := [width * height]*Vec{}

	nproc := runtime.NumCPU()
	c := height / nproc
	var wg sync.WaitGroup
	for I := 0; I < nproc; I++ {
		min := I * c
		max := (I + 1) * c
		if max > height {
			max = (I * c) + height%c
		}

		wg.Add(1)
		go (func(min, max int) {
			for j := min; j < max; j++ { // actual rendering loop
				for i := 0; i < width; i++ {
					dir_x := (float64(i) + 0.5) - width/2.0
					dir_y := -(float64(j) + 0.5) + height/2.0 // this flips the image at the same time
					dir_z := -height / (2.0 * math.Tan(fov/2.0))
					var hit Vec
					if sphere_trace(NewVec(0, 0, 3), NewVec(dir_x, dir_y, dir_z).Normalize(1), &hit) { // the camera is placed to (0,0,3) and it looks along the -z axis
						noise_level := (sphere_radius - hit.Norm()) / noise_amplitude
						light_dir := (NewVec(10, 10, 10).Sub(&hit)).Normalize(1) // one light is placed to (10,10,10)
						light_intensity := math.Max(0.4, light_dir.Dot(distance_field_normal(&hit)))
						framebuffer[i+j*width] = palette_fire((-.2 + noise_level) * 2).Mul(light_intensity)
					} else {
						framebuffer[i+j*width] = NewVec(0.2, 0.7, 0.8) // background color
					}
				}
			}
			wg.Done()
		})(min, max)
	}
	wg.Wait()

	b := &bytes.Buffer{}
	fmt.Fprintf(b, "P6\n%d %d\n255\n", width, height)
	for i := 0; i < height*width; i++ {
		b.WriteByte(byte(math.Max(0, math.Min(255, 255*framebuffer[i].x))))
		b.WriteByte(byte(math.Max(0, math.Min(255, 255*framebuffer[i].y))))
		b.WriteByte(byte(math.Max(0, math.Min(255, 255*framebuffer[i].z))))
	}

	if f, err := os.Create("./out-go.ppm"); err != nil {
		log.Print(err)
		return
	} else {
		defer f.Close()
		if _, err := f.Write(b.Bytes()); err != nil {
			log.Print(err)
		}
	}
}
