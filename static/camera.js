window.faceCamera = {
    isMobileCamera(label) {
        return /\b(phone|iphone|android|mobile|droidcam|iriun|epoccam|ivcam|continuity|phone link)\b/i.test(label);
    },

    cameraScore(label) {
        if (!label || this.isMobileCamera(label)) return -1;
        if (/\b(integrated|built[\s-]?in|internal|laptop|notebook)\b/i.test(label)) return 3;
        if (/\b(webcam|web cam|camera|hd user facing)\b/i.test(label)) return 2;
        if (/\b(virtual|obs|snap camera)\b/i.test(label)) return -1;
        return 0;
    },

    async getCameras() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return [];
        const devices = await navigator.mediaDevices.enumerateDevices();
        return devices.filter((device) => device.kind === "videoinput");
    },

    async updateCameraList(selector, selectedId) {
        if (!selector) return [];
        const cameras = await this.getCameras();
        const preferredCamera = cameras
            .filter((camera) => this.cameraScore(camera.label) >= 0)
            .sort((first, second) => this.cameraScore(second.label) - this.cameraScore(first.label))[0];

        selector.replaceChildren();
        const automaticOption = document.createElement("option");
        automaticOption.value = "";
        automaticOption.textContent = "Auto-select laptop webcam";
        selector.appendChild(automaticOption);

        cameras.forEach((camera, index) => {
            const option = document.createElement("option");
            option.value = camera.deviceId;
            option.textContent = camera.label || `Camera ${index + 1}`;
            selector.appendChild(option);
        });

        const selectedCamera = cameras.find((camera) => camera.deviceId === selectedId);
        selector.value = selectedCamera
            ? selectedCamera.deviceId
            : preferredCamera && this.cameraScore(preferredCamera.label) > 0
                ? preferredCamera.deviceId
                : "";
        return cameras;
    },

    async start(video, selector) {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("This browser cannot access a camera. Use a current browser over HTTPS or localhost.");
        }

        const requestedId = selector ? selector.value : "";
        const cameras = await this.getCameras();
        const namedCameras = cameras.filter((camera) => camera.label);
        const laptopCamera = namedCameras
            .filter((camera) => this.cameraScore(camera.label) > 0)
            .sort((first, second) => this.cameraScore(second.label) - this.cameraScore(first.label))[0];
        const onlyPhoneCameras = namedCameras.length > 0
            && namedCameras.every((camera) => this.isMobileCamera(camera.label));

        if (onlyPhoneCameras) {
            throw new Error("Only a phone camera is available. Connect or enable the laptop webcam, then refresh the camera list.");
        }
        if (!requestedId && !laptopCamera && cameras.length > 1) {
            throw new Error("Select your laptop webcam from the Camera Preview list before starting.");
        }

        const preferredId = requestedId || (laptopCamera && laptopCamera.deviceId) || "";
        const constraints = {
            audio: false,
            video: preferredId ? { deviceId: { exact: preferredId } } : true
        };
        let stream = await navigator.mediaDevices.getUserMedia(constraints);
        const previousStream = video.srcObject;
        let replacementStream = null;
        video.srcObject = stream;
        try {
            if (video.readyState < HTMLMediaElement.HAVE_METADATA) {
                await new Promise((resolve, reject) => {
                    video.addEventListener("loadedmetadata", resolve, { once: true });
                    video.addEventListener("error", () => reject(new Error("Could not load the camera preview.")), { once: true });
                });
            }
            await video.play();

            const refreshedCameras = await this.getCameras();
            const refreshedNamedCameras = refreshedCameras.filter((camera) => camera.label);
            const refreshedLaptopCamera = refreshedNamedCameras
                .filter((camera) => this.cameraScore(camera.label) > 0)
                .sort((first, second) => this.cameraScore(second.label) - this.cameraScore(first.label))[0];
            const refreshedOnlyPhoneCameras = refreshedNamedCameras.length > 0
                && refreshedNamedCameras.every((camera) => this.isMobileCamera(camera.label));
            if (!requestedId && refreshedOnlyPhoneCameras) {
                throw new Error("Only a phone camera is available. Connect or enable the laptop webcam, then refresh the camera list.");
            }

            const activeId = stream.getVideoTracks()[0].getSettings().deviceId || preferredId;
            const activeCamera = refreshedCameras.find((camera) => camera.deviceId === activeId);
            const activeIsPhoneCamera = activeCamera && this.isMobileCamera(activeCamera.label);
            if (activeIsPhoneCamera && !refreshedLaptopCamera) {
                throw new Error("The selected camera is a phone camera. Choose the laptop webcam from the Camera Preview list.");
            }
            if (refreshedLaptopCamera && activeId !== refreshedLaptopCamera.deviceId
                && (!requestedId || activeIsPhoneCamera)) {
                replacementStream = await navigator.mediaDevices.getUserMedia({
                    audio: false,
                    video: { deviceId: { exact: refreshedLaptopCamera.deviceId } }
                });
                video.srcObject = replacementStream;
                if (video.readyState < HTMLMediaElement.HAVE_METADATA) {
                    await new Promise((resolve, reject) => {
                        video.addEventListener("loadedmetadata", resolve, { once: true });
                        video.addEventListener("error", () => reject(new Error("Could not load the laptop webcam preview.")), { once: true });
                    });
                }
                await video.play();
                stream.getTracks().forEach((track) => track.stop());
                stream = replacementStream;
                replacementStream = null;
            }

            if (previousStream && previousStream !== stream) {
                previousStream.getTracks().forEach((track) => track.stop());
            }
            const activeCameraId = stream.getVideoTracks()[0].getSettings().deviceId || preferredId;
            await this.updateCameraList(selector, activeCameraId);
            return stream;
        } catch (error) {
            stream.getTracks().forEach((track) => track.stop());
            if (replacementStream) {
                replacementStream.getTracks().forEach((track) => track.stop());
            }
            video.srcObject = previousStream || null;
            throw error;
        }
    },

    stop(video) {
        if (!video.srcObject) return;
        video.srcObject.getTracks().forEach((track) => track.stop());
        video.srcObject = null;
    },

    captureFrame(video) {
        if (!video.videoWidth || !video.videoHeight) {
            return Promise.reject(new Error("Camera is not ready yet."));
        }

        const scale = Math.min(1, 640 / video.videoWidth, 480 / video.videoHeight);
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
        canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
        canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);

        return new Promise((resolve, reject) => {
            canvas.toBlob((blob) => {
                if (blob) {
                    resolve({ blob, width: canvas.width, height: canvas.height });
                } else {
                    reject(new Error("Could not capture a camera frame."));
                }
            }, "image/jpeg", 0.82);
        });
    }
};
