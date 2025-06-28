import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface ImageProcessingResponse {
  success: boolean;
  result?: string;
  histogram?: string;
  faces_count?: number;
  faces_anonymized?: number;
  width?: number;
  height?: number;
  channels?: number;
  image?: string;
}

@Injectable({
  providedIn: 'root'
})
export class ImageProcessingService {
  private apiUrl = 'http://localhost:8000';

  constructor(private http: HttpClient) { }

  uploadImage(file: File): Observable<ImageProcessingResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post<ImageProcessingResponse>(`${this.apiUrl}/upload`, formData);
  }

  convertToGrayscale(imageData: string): Observable<ImageProcessingResponse> {
    const formData = new FormData();
    formData.append('image_data', imageData);
    return this.http.post<ImageProcessingResponse>(`${this.apiUrl}/convert-grayscale`, formData);
  }

  generateHistogram(imageData: string): Observable<ImageProcessingResponse> {
    const formData = new FormData();
    formData.append('image_data', imageData);
    return this.http.post<ImageProcessingResponse>(`${this.apiUrl}/histogram`, formData);
  }

  flipImage(imageData: string, direction: string): Observable<ImageProcessingResponse> {
    const formData = new FormData();
    formData.append('image_data', imageData);
    formData.append('direction', direction);
    return this.http.post<ImageProcessingResponse>(`${this.apiUrl}/flip`, formData);
  }

  rotateImage(imageData: string, angle: number): Observable<ImageProcessingResponse> {
    const formData = new FormData();
    formData.append('image_data', imageData);
    formData.append('angle', angle.toString());
    return this.http.post<ImageProcessingResponse>(`${this.apiUrl}/rotate`, formData);
  }

  resizeImage(imageData: string, width: number, height: number): Observable<ImageProcessingResponse> {
    const formData = new FormData();
    formData.append('image_data', imageData);
    formData.append('width', width.toString());
    formData.append('height', height.toString());
    return this.http.post<ImageProcessingResponse>(`${this.apiUrl}/resize`, formData);
  }

  thresholdImage(imageData: string, method: string, thresholdValue?: number): Observable<ImageProcessingResponse> {
    const formData = new FormData();
    formData.append('image_data', imageData);
    formData.append('method', method);
    if (thresholdValue !== undefined) {
      formData.append('threshold_value', thresholdValue.toString());
    }
    return this.http.post<ImageProcessingResponse>(`${this.apiUrl}/threshold`, formData);
  }

  denoiseImage(imageData: string, method: string, kernelSize: number = 5): Observable<ImageProcessingResponse> {
    const formData = new FormData();
    formData.append('image_data', imageData);
    formData.append('method', method);
    formData.append('kernel_size', kernelSize.toString());
    return this.http.post<ImageProcessingResponse>(`${this.apiUrl}/denoise`, formData);
  }

  morphologyOperation(imageData: string, operation: string, kernelSize: number = 5): Observable<ImageProcessingResponse> {
    const formData = new FormData();
    formData.append('image_data', imageData);
    formData.append('operation', operation);
    formData.append('kernel_size', kernelSize.toString());
    return this.http.post<ImageProcessingResponse>(`${this.apiUrl}/morphology`, formData);
  }

  edgeDetection(imageData: string, method: string): Observable<ImageProcessingResponse> {
    const formData = new FormData();
    formData.append('image_data', imageData);
    formData.append('method', method);
    return this.http.post<ImageProcessingResponse>(`${this.apiUrl}/edge-detection`, formData);
  }

  superpixelSegmentation(imageData: string, nSegments: number = 100): Observable<ImageProcessingResponse> {
    const formData = new FormData();
    formData.append('image_data', imageData);
    formData.append('n_segments', nSegments.toString());
    return this.http.post<ImageProcessingResponse>(`${this.apiUrl}/superpixels`, formData);
  }

  faceDetection(imageData: string): Observable<ImageProcessingResponse> {
    const formData = new FormData();
    formData.append('image_data', imageData);
    return this.http.post<ImageProcessingResponse>(`${this.apiUrl}/face-detection`, formData);
  }

  faceAnonymization(imageData: string, method: string): Observable<ImageProcessingResponse> {
    const formData = new FormData();
    formData.append('image_data', imageData);
    formData.append('method', method);
    return this.http.post<ImageProcessingResponse>(`${this.apiUrl}/face-anonymization`, formData);
  }
}

