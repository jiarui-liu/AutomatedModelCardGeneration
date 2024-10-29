from huggingface_hub import HfApi
import json
import argparse
import time
api = HfApi()


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_file",  type=str) # json
    parser.add_argument("--type", type=str, choices=['model', 'dataset']) 
    parser.add_argument("--start_idx", type=int, default=0) 
    parser.add_argument("--end_idx", type=int)
    args = parser.parse_args() # exclusive end index
    return args

if __name__ == "__main__":
    args = arg_parse()

    pager_func = None
    if args.type == 'model':
        pager_func = api.list_models
    elif args.type == 'dataset':
        pager_func = api.list_datasets

    outer_idx = 0
    
    with open(args.out_file, "a") as out_f:
        try:
            for idx, i in enumerate(pager_func(sort="downloads", direction=-1, limit=args.end_idx)): # type: ignore
                outer_idx += 1
                if outer_idx % 100 == 0:
                    print(f"Processed {outer_idx} models...")
                if idx >= args.start_idx:
                    json.dump(
                        {
                            "modelId": i.modelId, # type: ignore
                            "likes": i.likes, # type: ignore
                            "downloads": i.downloads, # type: ignore
                            "tags": i.tags,
                            "pipeline_tag": i.pipeline_tag, # type: ignore
                            
                        } if args.type == 'model' else {
                            "datasetId": i.id, # type: ignore
                            "likes": i.likes, # type: ignore
                            "downloads": i.downloads, # type: ignore
                            "tags": i.tags,
                        },
                        out_f,
                        ensure_ascii=False
                    )
                    out_f.write("\n")
                    time.sleep(1)
        except Exception as e:
            print(f"Stop at index {outer_idx}", e)