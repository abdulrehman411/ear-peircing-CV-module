# Ear Piercing Alignment and Validation System

A standalone computer vision module for real-time ear detection, measurement, piercing detection, and alignment validation. This system provides REST APIs for integration with Node.js backend and React Native Expo frontend.

## Features

- **Real-Time Ear Detection**: Detect ears using MediaPipe Face Mesh landmarks
- **Ear Measurement**: Calculate ear length and width dimensions
- **Piercing Detection**: Automatically detect existing piercings using multiple methods
- **Digital Point Marking**: Mark piercing points on-screen with landmark references
- **Physical Point Validation**: Compare digital points with physical marks and provide feedback
- **Symmetry Mapping**: Automatically replicate piercing points to the second ear
- **Re-Scan Validation**: Support multiple validation iterations with history tracking

## Technology Stack

- **CV Framework**: MediaPipe (Face Mesh for ear landmarks)
- **API Framework**: FastAPI (Python)
- **Image Processing**: OpenCV, PIL/Pillow
- **Deployment**: Docker container
- **Integration**: REST APIs with JSON responses

## Project Structure

```
ear-piercing-cv-module/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── .env.example
├── src/
│   ├── main.py                 # FastAPI application entry
│   ├── config.py               # Configuration management
│   ├── models/                 # Data models
│   ├── services/               # CV processing services
│   ├── api/                    # API routes and schemas
│   ├── utils/                  # Utility functions
│   └── tests/                  # Test suite
```

## Installation

### Using Docker (Recommended)

1. Clone the repository
2. Copy `.env.example` to `.env` and configure as needed
3. Build and run with Docker Compose:

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`

### Local Development

1. Install Python 3.10+
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check
- **GET** `/api/v1/health`
- Returns service health status

### Ear Detection
- **POST** `/api/v1/detect-ear`
- Request: `{ "image": "base64_encoded_image" }`
- Response: Ear detection result with dimensions, landmarks, and bounding box

### Piercing Detection
- **POST** `/api/v1/detect-piercings`
- Request: `{ "image": "base64_encoded_image", "ear_landmarks": [...], "bounding_box": {...} }`
- Response: List of detected piercings with coordinates and types

### Mark Point
- **POST** `/api/v1/mark-point`
- Request: `{ "image": "base64_encoded_image", "point": { "x": 0.5, "y": 0.6 }, "ear_landmarks": [...], "bounding_box": {...} }`
- Response: Marked point with landmark references

### Validate Point
- **POST** `/api/v1/validate-point`
- Request: `{ "original_image": "base64", "rescan_image": "base64", "digital_point": {...}, "ear_landmarks": [...], "bounding_box": {...} }`
- Response: Validation result with offset and feedback

### Get Feedback
- **POST** `/api/v1/feedback`
- Request: `{ "rescan_image": "base64", "expected_point": {...}, "ear_landmarks": [...], "bounding_box": {...} }`
- Response: Real-time feedback for point adjustment

### Symmetry Map
- **POST** `/api/v1/symmetry-map`
- Request: `{ "ear1_image": "base64", "ear2_image": "base64", "ear1_point": {...}, ... }`
- Response: Mapped point for second ear with scale factors

### Re-Scan Validate
- **POST** `/api/v1/rescan-validate`
- Request: `{ "rescan_image": "base64", "original_point": {...}, "validation_history": [...], ... }`
- Response: Validation history with best result

## API Documentation

Once the server is running, interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Configuration

Configuration is managed through environment variables. See `.env.example` for available options:

- `VALIDATION_THRESHOLD`: Acceptable offset threshold (default: 0.01)
- `MEDIAPIPE_CONFIDENCE_THRESHOLD`: MediaPipe detection confidence (default: 0.5)
- `MAX_IMAGE_SIZE`: Maximum image dimension in pixels (default: 5000)
- `CORS_ORIGINS`: Allowed CORS origins (comma-separated)

## Integration with Node.js Backend

The CV module runs as a separate Docker service. Your Node.js backend can make HTTP requests to it:

```javascript
const axios = require('axios');

const cvModuleUrl = process.env.CV_MODULE_URL || 'http://localhost:8000';

async function detectEar(imageBase64) {
  const response = await axios.post(`${cvModuleUrl}/api/v1/detect-ear`, {
    image: imageBase64
  });
  return response.data;
}
```

## Integration with React Native Expo

### Camera Capture

Use `expo-camera` or `react-native-vision-camera` to capture images:

```javascript
import { Camera } from 'expo-camera';
import * as FileSystem from 'expo-file-system';

// Capture image
const photo = await camera.takePictureAsync({ base64: true });

// Send to backend (which forwards to CV module)
const response = await fetch('YOUR_BACKEND_URL/api/detect-ear', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ image: photo.base64 })
});
```

### Display Overlays

Display ear detection overlay and point marking UI:

```javascript
// Display landmarks
{landmarks.map((landmark, index) => (
  <View
    key={index}
    style={{
      position: 'absolute',
      left: landmark.x * width,
      top: landmark.y * height,
      width: 4,
      height: 4,
      backgroundColor: 'red',
      borderRadius: 2
    }}
  />
))}
```

## Testing

Run tests with pytest:

```bash
pytest src/tests/
```

## Performance Optimization

- MediaPipe model is cached after first initialization
- Image preprocessing is optimized for mobile devices
- Async API endpoints for better concurrency
- Configurable image size limits

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid input)
- `500`: Internal Server Error

Error responses follow this format:
```json
{
  "success": false,
  "error": "Error type",
  "message": "Error message",
  "details": {}
}
```

## Troubleshooting

### MediaPipe Not Detecting Ears
- Ensure good lighting conditions
- Check image quality and resolution
- Verify face is clearly visible in image
- Adjust `MEDIAPIPE_CONFIDENCE_THRESHOLD` if needed

### Mark Detection Issues
- Ensure marker color is blue or black
- Check that mark is clearly visible
- Verify ear region is properly extracted
- Adjust `mark_min_area` in configuration

### Performance Issues
- Reduce `MAX_IMAGE_SIZE` for faster processing
- Enable caching if not already enabled
- Consider using image compression before sending

## License

This project is part of a placement assessment.

## Support

For issues or questions, please refer to the API documentation at `/docs` endpoint.

