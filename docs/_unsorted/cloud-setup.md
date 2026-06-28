# Lambda AI


Machine: **A10**  at ~ 1.29$/h
Image: **Lambda Stack (22.04)**

**Lambda Stack (22.04).** The default image used on most ODC instances.
- _Includes:_ NVIDIA driver, CUDA toolkit, OFED, cuDNN, NCCL, Lambda Stack Python packages (PyTorch, TensorFlow, JAX), Docker, JupyterLab
- _Use if:_ You want something that "just works" and don't need deep configurability.



**Getting the instances**

Get your lambda API key and store is as follows:
```
chmod 600 ~/.secrets/lambda.api
```

We can request our running instances as follows:

```bash
curl -u "$(cat ~/.secrets/lambda.api):" \
https://cloud.lambdalabs.com/api/v1/instances | jq > instances.json && cat instances.json
```

```bash
curl -s -u "$(cat ~/.secrets/lambda.api):" \
  https://cloud.lambdalabs.com/api/v1/instances \
  | jq '.data[] | {region_name: .region.name, instance_type_name: .instance_type.name, ssh_key_names, file_system_names}' \
  > relaunch-spec.json && cat relaunch-spec.json
```

Note the trailing `:` stays outside the `$(...)` — it tells curl the password half is empty.
Lambda uses HTTP Basic with the key as the username and empty password, so `login` = your key, `password` = anything.

