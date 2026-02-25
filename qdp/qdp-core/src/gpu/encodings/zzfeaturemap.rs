//
// Licensed to the Apache Software Foundation (ASF) under one or more
// contributor license agreements.  See the NOTICE file distributed with
// this work for additional information regarding copyright ownership.
// The ASF licenses this file to You under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with
// the License.  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::QuantumEncoder;

use crate::error::{MahoutError, Result};
use crate::gpu::memory::GpuStateVector;

use cudarc::driver::CudaDevice;
use std::sync::Arc;

use num_complex::Complex64;

/// Entanglement pattern for ZZFeatureMap
#[derive(Clone, Copy)]
pub enum Entanglement {
    /// Nearest-neighbor entanglement:
    /// q0--q1--q2--q3...
    Linear,
}

/// ZZFeatureMap encoder (POC)
pub struct ZZFeatureMap {
    entanglement: Entanglement,
}

impl ZZFeatureMap {
    /// Create ZZFeatureMap with linear entanglement
    #[must_use]
    pub fn linear() -> Self {
        Self {
            entanglement: Entanglement::Linear,
        }
    }

    fn expected_data_len(&self, num_qubits: usize) -> usize {
        num_qubits
    }

    /// CPU reference implementation for validation
    pub fn cpu_reference_state(
        &self,
        data: &[f64],
        num_qubits: usize,
    ) -> Vec<Complex64> {
        let dim = 1 << num_qubits;
        let norm = 1.0 / (dim as f64).sqrt();

        let mut state = vec![Complex64::new(norm, 0.0); dim];

        for basis in 0..dim {
            let mut phase = 0.0;

            // Single-qubit Z rotations
            for q in 0..num_qubits {
                let bit = (basis >> q) & 1;
                let sign = if bit == 0 { -1.0 } else { 1.0 };
                phase += sign * data[q];
            }

            // Linear ZZ interactions
            if num_qubits > 1 {
                for q in 0..(num_qubits - 1) {
                    let bit_i = (basis >> q) & 1;
                    let bit_j = (basis >> (q + 1)) & 1;

                    let zi = if bit_i == 0 { 1.0 } else { -1.0 };
                    let zj = if bit_j == 0 { 1.0 } else { -1.0 };

                    phase += zi * zj * data[q] * data[q + 1];
                }
            }

            let complex_phase = Complex64::from_polar(1.0, phase);
            state[basis] *= complex_phase;
        }

        state
    }
}

impl QuantumEncoder for ZZFeatureMap {
    fn encode(
        &self,
        #[cfg(target_os = "linux")] _device: &Arc<CudaDevice>,
        #[cfg(not(target_os = "linux"))] _device: &Arc<CudaDevice>,
        data: &[f64],
        num_qubits: usize,
    ) -> Result<GpuStateVector> {
        self.validate_input(data, num_qubits)?;

        #[cfg(target_os = "linux")]
        {
            Err(MahoutError::Cuda(
                "ZZFeatureMap kernel not yet implemented (POC stage)"
                    .to_string(),
            ))
        }

        #[cfg(not(target_os = "linux"))]
        {
            Err(MahoutError::Cuda(
                "CUDA unavailable (non-Linux stub)".to_string(),
            ))
        }
    }

    fn validate_input(&self, data: &[f64], num_qubits: usize) -> Result<()> {
        if num_qubits == 0 {
            return Err(MahoutError::InvalidInput(
                "Number of qubits must be at least 1".to_string(),
            ));
        }

        if num_qubits > 30 {
            return Err(MahoutError::InvalidInput(format!(
                "Number of qubits {} exceeds practical limit of 30",
                num_qubits
            )));
        }

        if data.len() != self.expected_data_len(num_qubits) {
            return Err(MahoutError::InvalidInput(format!(
                "ZZFeatureMap expects {} values for {} qubits, got {}",
                num_qubits,
                num_qubits,
                data.len()
            )));
        }

        for (i, &val) in data.iter().enumerate() {
            if !val.is_finite() {
                return Err(MahoutError::InvalidInput(format!(
                    "Parameter at index {} must be finite, got {}",
                    i, val
                )));
            }
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "zzfeaturemap"
    }

    fn description(&self) -> &'static str {
        "ZZFeatureMap encoding with linear entanglement (POC)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_reference_normalized() {
        let encoder = ZZFeatureMap::linear();
        let data = vec![0.5, 1.0];
        let state = encoder.cpu_reference_state(&data, 2);

        assert_eq!(state.len(), 4);

        let norm: f64 = state.iter().map(|c| c.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-10);
    }
}