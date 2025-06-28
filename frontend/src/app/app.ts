import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';

// Angular Material imports
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatTabsModule } from '@angular/material/tabs';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatSliderModule } from '@angular/material/slider';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatSnackBarModule, MatSnackBar } from '@angular/material/snack-bar';
import { MatDividerModule } from '@angular/material/divider';
import { MatGridListModule } from '@angular/material/grid-list';

import { ImageProcessingService } from './services/image-processing.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    HttpClientModule,
    MatToolbarModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatTabsModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatSliderModule,
    MatProgressSpinnerModule,
    MatSnackBarModule,
    MatDividerModule,
    MatGridListModule
  ],
  templateUrl: './app.html',
  styleUrls: ['./app.scss']
})
export class AppComponent {
  title = 'Outil de Traitement d\'Images';
  
  // Image data
  originalImage: string | null = null;
  processedImage: string | null = null;
  histogramImage: string | null = null;
  
  // Loading states
  isLoading = false;
  isProcessing = false;
  
  // Form values
  rotationAngle = 0;
  resizeWidth = 500;
  resizeHeight = 500;
  thresholdValue = 128;
  kernelSize = 5;
  nSegments = 100;
  
  // Selected options
  flipDirection = 'horizontal';
  thresholdMethod = 'global';
  denoiseMethod = 'gaussian';
  morphologyOp = 'erosion';
  edgeMethod = 'canny';
  anonymizationMethod = 'blur';

  constructor(
    private imageService: ImageProcessingService,
    private snackBar: MatSnackBar
  ) {}

  onFileSelected(event: any): void {
    const file = event.target.files[0];
    if (file) {
      this.isLoading = true;
      this.imageService.uploadImage(file).subscribe({
        next: (response) => {
          if (response.success) {
            this.originalImage = 'data:image/png;base64,' + response.image;
            this.processedImage = null;
            this.histogramImage = null;
            this.resizeWidth = response.width || 500;
            this.resizeHeight = response.height || 500;
            this.showMessage('Image téléchargée avec succès');
          }
          this.isLoading = false;
        },
        error: (error) => {
          this.showMessage('Erreur lors du téléchargement de l\'image');
          this.isLoading = false;
        }
      });
    }
  }

  convertToGrayscale(): void {
    if (!this.originalImage) return;
    this.processImage(() => 
      this.imageService.convertToGrayscale(this.getImageData())
    );
  }

  generateHistogram(): void {
    if (!this.originalImage) return;
    this.isProcessing = true;
    this.imageService.generateHistogram(this.getImageData()).subscribe({
      next: (response) => {
        if (response.success) {
          this.histogramImage = 'data:image/png;base64,' + response.histogram;
          this.showMessage('Histogramme généré avec succès');
        }
        this.isProcessing = false;
      },
      error: (error) => {
        this.showMessage('Erreur lors de la génération de l\'histogramme');
        this.isProcessing = false;
      }
    });
  }

  flipImage(): void {
    if (!this.originalImage) return;
    this.processImage(() => 
      this.imageService.flipImage(this.getImageData(), this.flipDirection)
    );
  }

  rotateImage(): void {
    if (!this.originalImage) return;
    this.processImage(() => 
      this.imageService.rotateImage(this.getImageData(), this.rotationAngle)
    );
  }

  resizeImage(): void {
    if (!this.originalImage) return;
    this.processImage(() => 
      this.imageService.resizeImage(this.getImageData(), this.resizeWidth, this.resizeHeight)
    );
  }

  thresholdImage(): void {
    if (!this.originalImage) return;
    this.processImage(() => 
      this.imageService.thresholdImage(
        this.getImageData(), 
        this.thresholdMethod, 
        this.thresholdMethod === 'global' ? this.thresholdValue : undefined
      )
    );
  }

  denoiseImage(): void {
    if (!this.originalImage) return;
    this.processImage(() => 
      this.imageService.denoiseImage(this.getImageData(), this.denoiseMethod, this.kernelSize)
    );
  }

  morphologyOperation(): void {
    if (!this.originalImage) return;
    this.processImage(() => 
      this.imageService.morphologyOperation(this.getImageData(), this.morphologyOp, this.kernelSize)
    );
  }

  edgeDetection(): void {
    if (!this.originalImage) return;
    this.processImage(() => 
      this.imageService.edgeDetection(this.getImageData(), this.edgeMethod)
    );
  }

  superpixelSegmentation(): void {
    if (!this.originalImage) return;
    this.processImage(() => 
      this.imageService.superpixelSegmentation(this.getImageData(), this.nSegments)
    );
  }

  faceDetection(): void {
    if (!this.originalImage) return;
    this.processImage(() => 
      this.imageService.faceDetection(this.getImageData())
    );
  }

  faceAnonymization(): void {
    if (!this.originalImage) return;
    this.processImage(() => 
      this.imageService.faceAnonymization(this.getImageData(), this.anonymizationMethod)
    );
  }

  private processImage(operation: () => any): void {
    this.isProcessing = true;
    operation().subscribe({
      next: (response: any) => {
        if (response.success) {
          this.processedImage = 'data:image/png;base64,' + response.result;
          this.showMessage('Traitement appliqué avec succès');
        }
        this.isProcessing = false;
      },
      error: (error: any) => {
        this.showMessage('Erreur lors du traitement');
        this.isProcessing = false;
      }
    });
  }

  private getImageData(): string {
    if (!this.originalImage) return '';
    return this.originalImage.split(',')[1]; // Remove data:image/png;base64, prefix
  }

  downloadImage(): void {
    if (!this.processedImage) return;
    
    const link = document.createElement('a');
    link.href = this.processedImage;
    link.download = 'image_traitee.png';
    link.click();
  }

  resetImage(): void {
    this.processedImage = null;
    this.histogramImage = null;
  }

  private showMessage(message: string): void {
    this.snackBar.open(message, 'Fermer', {
      duration: 3000,
      horizontalPosition: 'center',
      verticalPosition: 'bottom'
    });
  }
}

