# Integration Guide

This guide provides detailed instructions for integrating the Ear Piercing CV Module with your Node.js backend and React Native Expo frontend.

## Table of Contents

1. [Node.js Backend Integration](#nodejs-backend-integration)
2. [React Native Expo Integration](#react-native-expo-integration)
3. [API Usage Examples](#api-usage-examples)
4. [Error Handling](#error-handling)
5. [Best Practices](#best-practices)

## Node.js Backend Integration

### Setup

1. Install HTTP client library (axios recommended):

```bash
npm install axios
```

2. Configure CV module URL:

```javascript
// config.js
module.exports = {
  cvModuleUrl: process.env.CV_MODULE_URL || 'http://localhost:8000',
  cvModuleTimeout: 30000 // 30 seconds
};
```

### Service Layer

Create a service to communicate with the CV module:

```javascript
// services/cvModuleService.js
const axios = require('axios');
const config = require('../config');

class CVModuleService {
  constructor() {
    this.baseURL = `${config.cvModuleUrl}/api/v1`;
    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: config.cvModuleTimeout,
      headers: { 'Content-Type': 'application/json' }
    });
  }

  async detectEar(imageBase64) {
    try {
      const response = await this.client.post('/detect-ear', {
        image: imageBase64
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async detectPiercings(imageBase64, earLandmarks, boundingBox) {
    try {
      const response = await this.client.post('/detect-piercings', {
        image: imageBase64,
        ear_landmarks: earLandmarks,
        bounding_box: boundingBox
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async markPoint(imageBase64, point, earLandmarks, boundingBox) {
    try {
      const response = await this.client.post('/mark-point', {
        image: imageBase64,
        point: point,
        ear_landmarks: earLandmarks,
        bounding_box: boundingBox
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async validatePoint(originalImage, rescanImage, digitalPoint, earLandmarks, boundingBox) {
    try {
      const response = await this.client.post('/validate-point', {
        original_image: originalImage,
        rescan_image: rescanImage,
        digital_point: digitalPoint,
        ear_landmarks: earLandmarks,
        bounding_box: boundingBox
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getFeedback(rescanImage, expectedPoint, earLandmarks, boundingBox) {
    try {
      const response = await this.client.post('/feedback', {
        rescan_image: rescanImage,
        expected_point: expectedPoint,
        ear_landmarks: earLandmarks,
        bounding_box: boundingBox
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async symmetryMap(ear1Image, ear2Image, ear1Point, ear1Landmarks, ear2Landmarks, ear1BBox, ear2BBox) {
    try {
      const response = await this.client.post('/symmetry-map', {
        ear1_image: ear1Image,
        ear2_image: ear2Image,
        ear1_point: ear1Point,
        ear1_landmarks: ear1Landmarks,
        ear2_landmarks: ear2Landmarks,
        ear1_bounding_box: ear1BBox,
        ear2_bounding_box: ear2BBox
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async rescanValidate(rescanImage, originalPoint, validationHistory, earLandmarks, boundingBox) {
    try {
      const response = await this.client.post('/rescan-validate', {
        rescan_image: rescanImage,
        original_point: originalPoint,
        validation_history: validationHistory,
        ear_landmarks: earLandmarks,
        bounding_box: boundingBox
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  handleError(error) {
    if (error.response) {
      // CV module returned an error
      return new Error(`CV Module Error: ${error.response.data.message || error.response.data.error}`);
    } else if (error.request) {
      // Request made but no response
      return new Error('CV Module is not responding');
    } else {
      // Error in request setup
      return new Error(`Request Error: ${error.message}`);
    }
  }
}

module.exports = new CVModuleService();
```

### API Routes

Create Express routes that use the CV service:

```javascript
// routes/ear.js
const express = require('express');
const router = express.Router();
const cvModule = require('../services/cvModuleService');

router.post('/detect-ear', async (req, res) => {
  try {
    const { image } = req.body;
    const result = await cvModule.detectEar(image);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Add other routes similarly...

module.exports = router;
```

## React Native Expo Integration

### Setup

1. Install required packages:

```bash
expo install expo-camera expo-file-system
```

### Camera Component

```javascript
// components/EarScanner.js
import React, { useState, useRef } from 'react';
import { View, StyleSheet, TouchableOpacity, Text, Image } from 'react-native';
import { Camera } from 'expo-camera';
import * as FileSystem from 'expo-file-system';

export default function EarScanner({ onImageCaptured }) {
  const [hasPermission, setHasPermission] = useState(null);
  const [type, setType] = useState(Camera.Constants.Type.front);
  const cameraRef = useRef(null);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);

  const takePicture = async () => {
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync({
        base64: true,
        quality: 0.8
      });
      onImageCaptured(photo.base64);
    }
  };

  if (hasPermission === null) {
    return <View />;
  }
  if (hasPermission === false) {
    return <Text>No access to camera</Text>;
  }

  return (
    <View style={styles.container}>
      <Camera style={styles.camera} type={type} ref={cameraRef}>
        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.button} onPress={takePicture}>
            <Text style={styles.text}>Capture</Text>
          </TouchableOpacity>
        </View>
      </Camera>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  camera: { flex: 1 },
  buttonContainer: {
    flex: 1,
    backgroundColor: 'transparent',
    flexDirection: 'row',
    margin: 20,
    justifyContent: 'center',
    alignItems: 'flex-end'
  },
  button: {
    backgroundColor: '#fff',
    padding: 15,
    borderRadius: 50
  },
  text: { fontSize: 18, color: 'black' }
});
```

### API Service

```javascript
// services/api.js
const API_URL = 'YOUR_BACKEND_URL';

export const detectEar = async (imageBase64) => {
  const response = await fetch(`${API_URL}/api/ear/detect-ear`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: imageBase64 })
  });
  return response.json();
};

export const markPoint = async (imageBase64, point, earLandmarks, boundingBox) => {
  const response = await fetch(`${API_URL}/api/ear/mark-point`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image: imageBase64,
      point: point,
      ear_landmarks: earLandmarks,
      bounding_box: boundingBox
    })
  });
  return response.json();
};

// Add other API functions...
```

### Usage Example

```javascript
// screens/EarPiercingScreen.js
import React, { useState } from 'react';
import { View, Text } from 'react-native';
import EarScanner from '../components/EarScanner';
import { detectEar, markPoint } from '../services/api';

export default function EarPiercingScreen() {
  const [earData, setEarData] = useState(null);
  const [selectedPoint, setSelectedPoint] = useState(null);

  const handleImageCaptured = async (imageBase64) => {
    try {
      const result = await detectEar(imageBase64);
      if (result.ear_detected) {
        setEarData(result);
      }
    } catch (error) {
      console.error('Ear detection failed:', error);
    }
  };

  const handlePointSelected = async (point) => {
    if (earData) {
      try {
        const result = await markPoint(
          imageBase64,
          point,
          earData.landmarks,
          earData.bounding_box
        );
        setSelectedPoint(result.point);
      } catch (error) {
        console.error('Point marking failed:', error);
      }
    }
  };

  return (
    <View>
      <EarScanner onImageCaptured={handleImageCaptured} />
      {earData && (
        <View>
          <Text>Ear detected: {earData.confidence}</Text>
          <Text>Dimensions: {earData.dimensions.length} x {earData.dimensions.width}</Text>
        </View>
      )}
    </View>
  );
}
```

## API Usage Examples

### Complete Workflow

```javascript
// 1. Detect ear
const earResult = await cvModule.detectEar(imageBase64);
if (!earResult.ear_detected) {
  throw new Error('No ear detected');
}

// 2. Detect existing piercings
const piercings = await cvModule.detectPiercings(
  imageBase64,
  earResult.landmarks,
  earResult.bounding_box
);

// 3. Mark digital point
const markedPoint = await cvModule.markPoint(
  imageBase64,
  { x: 0.5, y: 0.6 },
  earResult.landmarks,
  earResult.bounding_box
);

// 4. Validate physical mark
const validation = await cvModule.validatePoint(
  imageBase64,
  rescanImageBase64,
  markedPoint.point,
  earResult.landmarks,
  earResult.bounding_box
);

// 5. Get feedback
if (!validation.valid) {
  const feedback = await cvModule.getFeedback(
    rescanImageBase64,
    markedPoint.point,
    earResult.landmarks,
    earResult.bounding_box
  );
  console.log(feedback.message);
}
```

## Error Handling

Always handle errors gracefully:

```javascript
try {
  const result = await cvModule.detectEar(imageBase64);
  // Process result
} catch (error) {
  if (error.message.includes('not responding')) {
    // Retry or show offline message
  } else if (error.message.includes('CV Module Error')) {
    // Handle CV module specific error
  } else {
    // Handle other errors
  }
}
```

## Best Practices

1. **Image Compression**: Compress images before sending to reduce payload size
2. **Caching**: Cache ear detection results to avoid redundant API calls
3. **Error Retry**: Implement retry logic for network errors
4. **Loading States**: Show loading indicators during API calls
5. **Validation**: Validate image format and size before sending
6. **Timeout Handling**: Set appropriate timeouts for API calls
7. **Offline Support**: Handle offline scenarios gracefully

