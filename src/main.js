import * as THREE from 'three';

export function initWaterSimulation(containerElement = document.body) {
    const resolution = 256;
    const resSq = resolution * resolution;

    // 物理计算缓冲
    let grid = new Float32Array(resSq);
    let prev_grid = new Float32Array(resSq);
    let next_grid = new Float32Array(resSq);

    // 渲染专用缓冲
    let smooth_grid = new Float32Array(resSq);
    let rgb_data = new Uint8Array(resSq * 4); // RGBA 格式

    // 核心卷积核
    const kernel = [
        0.05, 0.20, 0.05,
        0.20, 0.00, 0.20,
        0.05, 0.20, 0.05
    ];
    const blur_kernel = [
        0.05, 0.10, 0.05,
        0.10, 0.40, 0.10,
        0.05, 0.10, 0.05
    ];

    function drop_water(cx, cy) {
        const radius = 3.0;
        const r22 = 2 * radius * radius;
        for (let y = 0; y < resolution; y++) {
            for (let x = 0; x < resolution; x++) {
                let dist_sq = (x - cx) ** 2 + (y - cy) ** 2;
                let blob = Math.exp(-dist_sq / r22);
                grid[y * resolution + x] += blob * 5.0;
            }
        }
    }

    function conv2d_3x3(input, output, k) {
        for (let y = 1; y < resolution - 1; y++) {
            for (let x = 1; x < resolution - 1; x++) {
                let sum = 0;
                sum += input[(y - 1) * resolution + (x - 1)] * k[0];
                sum += input[(y - 1) * resolution + x] * k[1];
                sum += input[(y - 1) * resolution + (x + 1)] * k[2];
                sum += input[y * resolution + (x - 1)] * k[3];
                sum += input[y * resolution + x] * k[4];
                sum += input[y * resolution + (x + 1)] * k[5];
                sum += input[(y + 1) * resolution + (x - 1)] * k[6];
                sum += input[(y + 1) * resolution + x] * k[7];
                sum += input[(y + 1) * resolution + (x + 1)] * k[8];
                output[y * resolution + x] = sum;
            }
        }
    }

    function spread() {
        let temp1 = new Float32Array(resSq);
        let temp2 = new Float32Array(resSq);

        conv2d_3x3(grid, temp1, kernel);

        for (let i = 0; i < resSq; i++) {
            next_grid[i] = (temp1[i] * 2.0 - prev_grid[i]) * 0.98;
        }

        conv2d_3x3(next_grid, temp2, blur_kernel);

        const viscosity = 0.15;
        for (let i = 0; i < resSq; i++) {
            next_grid[i] = next_grid[i] * (1.0 - viscosity) + temp2[i] * viscosity;
        }

        prev_grid.set(grid);
        grid.set(next_grid);
    }

    function smooth_5x5() {
        const weight = 1.0 / 25.0;
        for (let y = 2; y < resolution - 2; y++) {
            for (let x = 2; x < resolution - 2; x++) {
                let sum = 0;
                for (let ky = -2; ky <= 2; ky++) {
                    for (let kx = -2; kx <= 2; kx++) {
                        sum += grid[(y + ky) * resolution + (x + kx)] * weight;
                    }
                }
                smooth_grid[y * resolution + x] = sum;
            }
        }
    }

    const l = [-0.5, -0.5, 0.7];
    const l_mag = Math.sqrt(l[0] * l[0] + l[1] * l[1] + l[2] * l[2]);
    const light_dir = [l[0] / l_mag, l[1] / l_mag, l[2] / l_mag];
    const view_dir = [0.0, 0.0, 1.0];
    
    const h_vec = [light_dir[0] + view_dir[0], light_dir[1] + view_dir[1], light_dir[2] + view_dir[2]];
    const h_mag = Math.sqrt(h_vec[0] * h_vec[0] + h_vec[1] * h_vec[1] + h_vec[2] * h_vec[2]);
    const half_vec = [h_vec[0] / h_mag, h_vec[1] / h_mag, h_vec[2] / h_mag];

    function render_modern() {
        smooth_5x5();

        const slope_strength = 25.0;
        const refract_scale = 0.15;

        for (let y = 2; y < resolution - 2; y++) {
            for (let x = 2; x < resolution - 2; x++) {
                // 中心差分获取梯度
                let dx = (smooth_grid[y * resolution + (x + 1)] - smooth_grid[y * resolution + (x - 1)]) / 2.0;
                let dy = (smooth_grid[(y + 1) * resolution + x] - smooth_grid[(y - 1) * resolution + x]) / 2.0;
                
                dx *= slope_strength;
                dy *= slope_strength;

                let mag = Math.sqrt(dx * dx + dy * dy + 1.0);
                let nx = -dx / mag, ny = -dy / mag, nz = 1.0 / mag;

                let u = x / (resolution - 1);
                let v = y / (resolution - 1);

                let rx = Math.max(0, Math.min(1, u + nx * refract_scale));
                let ry = Math.max(0, Math.min(1, v + ny * refract_scale));

                let base_r = rx * 0.2 + 0.7;
                let base_g = 0.75;
                let base_b = ry * 0.2 + 0.8;

                let dot = nx * half_vec[0] + ny * half_vec[1] + nz * half_vec[2];
                let spec_dot = Math.max(0, Math.min(1, dot));
                let specular = Math.pow(spec_dot, 64.0);

                let ao = Math.max(0.8, Math.min(1.0, nz));

                let raw_r = base_r * ao + specular * 0.8;
                let raw_g = base_g * ao + specular * 0.8;
                let raw_b = base_b * ao + specular * 0.9;

                let idx = (y * resolution + x) * 4;
                rgb_data[idx]     = Math.max(0, Math.min(255, Math.tanh(raw_r) * 255));
                rgb_data[idx + 1] = Math.max(0, Math.min(255, Math.tanh(raw_g) * 255));
                rgb_data[idx + 2] = Math.max(0, Math.min(255, Math.tanh(raw_b) * 255));
                rgb_data[idx + 3] = 255;
            }
        }
    }

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#000000'); // 保持黑底以凸显水面

    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 5, 8);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    containerElement.appendChild(renderer.domElement);

    const dataTexture = new THREE.DataTexture(rgb_data, resolution, resolution, THREE.RGBAFormat);
    dataTexture.needsUpdate = true;

    const material = new THREE.MeshBasicMaterial({ map: dataTexture, side: THREE.DoubleSide });
    const geometry = new THREE.PlaneGeometry(10, 10);
    const waterMesh = new THREE.Mesh(geometry, material);
    
    waterMesh.rotation.x = -Math.PI / 2;
    scene.add(waterMesh);

    let animationFrameId;

    function animate() {
        animationFrameId = requestAnimationFrame(animate);

        if (Math.random() < 0.05) {
            let rx = Math.floor(Math.random() * (resolution - 10) + 5);
            let ry = Math.floor(Math.random() * (resolution - 10) + 5);
            drop_water(rx, ry);
        }

        spread();
        render_modern();

        dataTexture.needsUpdate = true;

        // 缓慢自转
        waterMesh.rotation.z += 0.003;

        renderer.render(scene, camera);
    }

    function onWindowResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }
    window.addEventListener('resize', onWindowResize);

    animate();

    return function destroy() {
        cancelAnimationFrame(animationFrameId);
        window.removeEventListener('resize', onWindowResize);
        containerElement.removeChild(renderer.domElement);
        geometry.dispose();
        material.dispose();
        dataTexture.dispose();
        renderer.dispose();
    };
}

const destroyWater = initWaterSimulation(document.body);