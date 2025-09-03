"""
Script to render zoom-in on Hilbert-space animation
"""

import math
import os
import sys

import glm
import moderngl
import numpy as np
import pygame
from hilbertcurve.hilbertcurve import HilbertCurve
import ipaddress

class Scene:

    def __init__(self, filepath, target):
        
        hc = HilbertCurve(p = 16, n = 2) # 2^np points so with n = 2 we need p = 16

        # Load IP addresses
        cachepath = filepath + ".cache.npy"
        if os.path.exists(cachepath):
            print("Loading cache...")
            locs = np.load(cachepath)
        else:
            print(f"Loading from {filepath}...")
            addrs = np.array([int(ipaddress.ip_address(x)) for x in  np.loadtxt(filepath, dtype = str)])
            print("Loaded addresses.")
            
            
            locs = np.array(hc.points_from_distances(addrs)).astype('f4')
            
            # Normalize to [-1, 1]
            locs /= 2 ** (16 - 1)
            locs -= 1
            print("Computed distances.")
            
            np.save(cachepath, locs)

        # Setup graphics
        
        self.ctx = moderngl.get_context()

        self.program = self.ctx.program(
            vertex_shader='''
                #version 330 core

                uniform float zoom = 1.0;
                uniform float aspect = 1.0;
                uniform vec2 target = vec2(0,0);
            
                layout (location = 0) in vec2 in_vertex;

                void main() {
                    gl_Position = vec4((in_vertex - target) * zoom, 0.0, 1.0);
                    gl_Position.x /= aspect;
                }
            ''',
            fragment_shader='''
                #version 330 core

                void main() {
                    gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
                }
            ''',
        )

        self.vbo = self.ctx.buffer(locs.tobytes())
        self.vao = self.ctx.vertex_array(
            self.program, [(self.vbo, '2f', 'in_vertex')],
            mode = self.ctx.POINTS
        )

        self.start_time = None
        self.zoom = 1
        self.target = np.array(hc.point_from_distance(int(ipaddress.ip_address(target)))).astype('f4')
        self.target /= 2 ** (16 - 1)
        self.target -= 1

        self.duration = 60 * 1000

        _, _, sw, sh = self.ctx.viewport
        self.aspect = sw / sh
        self.program['aspect'] = self.aspect
        

    def render(self):
        if self.start_time is None:
            self.start_time = pygame.time.get_ticks()
            
        self.ctx.clear()
        self.ctx.point_size = 1
        self.program['zoom'] = self.zoom
        self.program['target'] = self.target
        self.vao.render()

        cur_time = (pygame.time.get_ticks() - self.start_time) / self.duration
        self.zoom = 2 ** (28 * cur_time - 4)
        
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <file with list of source IPv4 addresses> <target address to zoom in on>")
        sys.exit()

    os.environ['SDL_WINDOWS_DPI_AWARENESS'] = 'permonitorv2'

    pygame.init()
    pygame.display.set_mode((1920, 1080), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)
        
    scene = Scene(sys.argv[1], sys.argv[2])

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    
        scene.render()
    
        pygame.display.flip()
