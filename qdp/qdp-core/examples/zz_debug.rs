use qdp_core::gpu::encodings::zzfeaturemap::ZZFeatureMap;

fn main() {
    let encoder = ZZFeatureMap::linear();
    let data = vec![0.5, 1.0];

    let state = encoder.cpu_reference_state(&data, 2);

    for (i, amp) in state.iter().enumerate() {
        println!("{}: {:.8} + {:.8}i", i, amp.re, amp.im);
    }
}
