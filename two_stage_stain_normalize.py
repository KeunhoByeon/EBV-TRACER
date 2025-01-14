import argparse

from src import TwoStageStainNormalizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EBV-TRACER Two-stage Stain Normalization')
    parser.add_argument('--target_svs_dir', default="", type=str)
    parser.add_argument('--input_svs_dir', default="", type=str)
    parser.add_argument('--input_patch_dir', default="", type=str)
    parser.add_argument('--output_patch_dir', default="", type=str)
    parser.add_argument('--output_thumbnail_dir', default="", type=str)
    parser.add_argument('--thumbnail_size', default=8192, type=int)
    parser.add_argument('--n_jobs', default=8, type=int)
    args = parser.parse_args()

    two_stage_normalizer = TwoStageStainNormalizer(
        target_svs_dir=args.target_svs_dir,
        input_svs_dir=args.input_svs_dir,
        input_patch_dir=args.input_patch_dir,
        output_patch_dir=args.output_patch_dir,
        output_thumbnail_dir=args.output_thumbnail_dir,
        thumbnail_size=args.thumbnail_size, n_jobs=args.n_jobs
    )
    two_stage_normalizer.two_stage_normalize()
