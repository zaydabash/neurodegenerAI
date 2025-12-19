# NeuroDegenerAI Model Card

## Model Details

**Model Name**: NeuroDegenerAI Early Detection Model
**Version**: 0.1.0
**Date**: December 2024
**Model Type**: Ensemble (LightGBM + CNN)
**Task**: Binary Classification (Normal vs. Cognitive Impairment)

## Intended Use

### Primary Use Cases
- **Research**: Early detection of neurodegenerative patterns in research settings
- **Screening**: Preliminary assessment of cognitive health
- **Clinical Decision Support**: Assisting healthcare professionals with additional insights

### Out-of-Scope Uses
- **Clinical Diagnosis**: This model is NOT intended for clinical diagnosis
- **Treatment Decisions**: Should not be used to make treatment decisions
- **Standalone Medical Device**: Not a replacement for clinical evaluation

## Training Data

### Data Sources
- **Primary**: ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset
- **Demo Mode**: Synthetic data with realistic distributions
- **Features**: Demographics, cognitive scores, biomarkers, MRI scans

### Data Characteristics
- **Sample Size**: 1,000+ subjects (demo mode), variable (real data)
- **Features**:
  - Demographics: Age, sex, education
  - Genetic: APOE4 status
  - Cognitive: MMSE, CDR scores
  - Biomarkers: Amyloid beta, tau, phosphorylated tau
  - Imaging: Hippocampal volume, cortical thickness, white matter hyperintensities
- **Labels**: Binary classification (Normal=0, Cognitive Impairment=1)

### Data Preprocessing
- Missing value imputation using median/mode
- Feature engineering (interactions, ratios, bins)
- Standardization for tabular features
- MRI preprocessing (resizing, normalization, slice extraction)

## Model Architecture

### Tabular Model (LightGBM)
- **Algorithm**: LightGBM Gradient Boosting
- **Parameters**:
  - num_leaves: 31
  - learning_rate: 0.05
  - n_estimators: 1000
  - early_stopping_rounds: 50
- **Features**: 50+ engineered features
- **Calibration**: Isotonic regression

### CNN Model (MRI)
- **Architecture**: ResNet18 adapted for 2.5D MRI slices
- **Input**: 64x64x64 MRI volumes → 16 slices
- **Layers**: 4 convolutional blocks + global average pooling
- **Output**: Binary classification
- **Pretraining**: ImageNet weights (adapted)

### Ensemble Method
- **Combination**: Weighted average (Tabular: 60%, CNN: 40%)
- **Calibration**: Platt scaling for probability calibration

## Performance Metrics

### Validation Performance (Demo Mode)
- **Accuracy**: 0.847
- **Precision**: 0.823
- **Recall**: 0.856
- **F1-Score**: 0.839
- **ROC-AUC**: 0.923
- **PR-AUC**: 0.887

### Calibration
- **Brier Score**: 0.156
- **ECE**: 0.089 (Expected Calibration Error)

## Limitations

### Model Limitations
1. **Training Data**: Performance may not generalize to all populations
2. **Feature Dependencies**: Requires specific biomarker and imaging data
3. **Temporal Dynamics**: Static model, doesn't account for disease progression
4. **Interpretability**: Complex ensemble may be difficult to interpret

### Data Limitations
1. **Sample Bias**: ADNI participants may not represent general population
2. **Missing Data**: Some biomarkers may not be available in clinical settings
3. **Imaging Quality**: MRI quality affects CNN performance
4. **Temporal Gaps**: Cross-sectional data may miss important temporal patterns

## Bias and Fairness

### Potential Biases
- **Age Bias**: Model trained primarily on elderly populations
- **Education Bias**: Higher education levels in ADNI dataset
- **Geographic Bias**: Primarily US-based participants
- **Socioeconomic Bias**: Research participants may not represent general population

### Mitigation Strategies
- Regular bias audits using demographic subgroups
- Balanced sampling in training data
- Fairness constraints in model training
- Continuous monitoring of performance across groups

## Interpretability

### Tabular Model
- **SHAP Values**: Feature importance and contributions
- **Feature Importance**: Top 20 most important features
- **Partial Dependence Plots**: Individual feature effects

### CNN Model
- **Grad-CAM**: Visual attention maps highlighting important brain regions
- **Integrated Gradients**: Attribution to input pixels
- **Layer-wise Relevance Propagation**: Neuron-level explanations

## Deployment Considerations

### System Requirements
- **Memory**: 2GB RAM minimum, 4GB recommended
- **Storage**: 500MB for models and dependencies
- **CPU**: 2 cores minimum for real-time inference
- **GPU**: Optional but recommended for CNN inference

### API Specifications
- **Endpoint**: `/predict/tabular`, `/predict/mri`, `/predict/ensemble`
- **Input Format**: JSON with required fields
- **Output Format**: JSON with predictions and explanations
- **Response Time**: <2 seconds for tabular, <10 seconds for MRI

## Monitoring and Maintenance

### Performance Monitoring
- Regular accuracy assessments on new data
- Drift detection for input distributions
- Calibration monitoring over time
- A/B testing for model updates

### Update Schedule
- **Quarterly**: Performance review and bias audits
- **Annually**: Model retraining with new data
- **As Needed**: Emergency updates for critical issues

## Ethical Considerations

### Clinical Use Disclaimer
**IMPORTANT**: This model is for research and screening purposes only. It should NOT be used for:
- Clinical diagnosis of Alzheimer's disease or other dementias
- Treatment decisions without clinical supervision
- Replacing clinical judgment or comprehensive evaluation
- Standalone medical decision-making

### Privacy and Security
- **Data Privacy**: No patient data is stored permanently
- **Security**: HTTPS encryption for all API communications
- **Access Control**: Authentication required for production use
- **Audit Logging**: All predictions logged for monitoring

### Regulatory Compliance
- **FDA Status**: Not FDA-approved medical device
- **Research Use**: Intended for research and educational purposes
- **Clinical Trials**: May be used in research studies with proper IRB approval
- **International**: Compliance with local medical device regulations required

## Contact Information

**Development Team**: Neuro-Trends Team
**Repository**: [GitHub Repository URL]
**Documentation**: [Documentation URL]
**Support**: [Support Contact]

## Citation

If you use this model in your research, please cite:

```bibtex
@software{neurodegenerai2024,
  title={NeuroDegenerAI: Early Neurodegenerative Pattern Detection},
  author={Neuro-Trends Team},
  year={2024},
  url={https://github.com/neuro-trends-suite},
  version={0.1.0}
}
```

## License

This model is released under the MIT License. See LICENSE file for details.

---

**Last Updated**: December 2024
**Next Review**: March 2025
