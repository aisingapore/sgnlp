local build_and_push_staging(module_name, image_name) = {
  image: "registry.aisingapore.net/sg-nlp/sg-nlp-runner:latest",
  stage: "build_and_push_staging",
  tags: "on-prem",
  when: "manual",
  script: [
    "echo 'Logging in to AISG Docker Registry...'",
    "echo $STG_REGISTRY_PASSWORD | docker login registry.aisingapore.net -u $STG_DOCKER_USER --password-stdin",
    "echo 'Building and pushing image...'",
    "docker build --no-cache -t %s -f demo_api/%s/Dockerfile demo_api/" % [module_name, module_name],
    "docker tag %s registry.aisingapore.net/sg-nlp/%s:latest" % [module_name, image_name],
    "docker push registry.aisingapore.net/sg-nlp/%s:latest" % image_name,
  ],
};

local build_and_push_docs_staging() = {
  image: "python:3.8.11-slim",
  stage: "build_and_push_staging",
  tags: "on-prem",
  when: "manual",
  script: [
    "echo 'Building Sphinx docs'",
    "cd docs",
    "echo 'Installing docker'",
    "apt-get update",
    "apt-get -y install apt-transport-https ca-certificates curl gnupg lsb-release",
    "curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg",
    'echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null',
    "apt-get update",
    "apt-get -y install docker-ce docker-ce-cli containerd.io",
    "echo 'Installing docs dependencies'",
    "apt-get -y install build-essential",
    "pip install --upgrade pip",
    "pip install -r requirements.txt -r ../requirements_extra.txt -e ../.",
    "sphinx-build -b html source build",
    "echo 'Building SG-NLP Documentation'",
    "echo $STG_REGISTRY_PASSWORD | docker login registry.aisingapore.net -u $STG_DOCKER_USER --password-stdin",
    "docker build -t docs -f Dockerfile .",
    "docker tag docs registry.aisingapore.net/sg-nlp/docs:latest",
    "docker push registry.aisingapore.net/sg-nlp/docs:latest",
  ]
};

local retag_and_push_production(module_name, image_name) = {
  image: "registry.aisingapore.net/sg-nlp/sg-nlp-runner:latest",
  stage: "retag_and_push_production",
  tags: "on-prem",
  only: {
    refs: ["main"]
  },
  needs: ["%s_restart_kubernetes_staging" % module_name],
  when: "manual",
  script: [
    "echo 'Logging in to AISG Docker Registry...'",
    "echo $STG_REGISTRY_PASSWORD | docker login registry.aisingapore.net -u $STG_DOCKER_USER --password-stdin",
    "docker pull registry.aisingapore.net/sg-nlp/%s:latest" % [image_name],
    "echo 'Logging in to GKE Registry'",
    "cat $GCR_KEYFILE | docker login -u _json_key --password-stdin https://asia.gcr.io",
    "echo 'Retagging and pushing image...'",
    "docker tag registry.aisingapore.net/sg-nlp/%s:latest asia.gcr.io/infra-public-hosting/sgnlp/%s:latest" % [image_name, image_name],
    "docker push asia.gcr.io/infra-public-hosting/sgnlp/%s:latest" % image_name,
  ],
};

local restart_kubernetes_staging(module_name, deployment_name) = {
  image: "registry.aisingapore.net/sea-core-nlp/seacorenlp-runner:latest",
  stage: "restart_kubernetes_staging",
  tags: "on-prem",
  when: "manual",
  needs: ["%s_build_and_push_staging" % module_name],
  script: [
    "echo 'Restarting pods...'",
    "export KUBECONFIG=$STG_KUBE_CONFIG",
    "kubectl rollout restart deployment/%s -n sg-nlp-revamp" % deployment_name
  ]
};

local restart_kubernetes_production(module_name, deployment_name) = {
  image: "registry.aisingapore.net/sea-core-nlp/seacorenlp-runner:latest",
  stage: "restart_kubernetes_production",
  tags: "on-prem",
  only: {
    refs: ["main"]
  },
  when: "manual",
  needs: ["%s_retag_and_push_production" % module_name],
  script: [
    "echo 'Restarting pods...'",
    "export KUBECONFIG=$PROD_GKE_KUBE_CONFIG",
    "kubectl rollout restart deployment/%s -n sgnlp" % deployment_name
  ]
};

local api_names = {
  "emotion_entailment": {
    module_name: "emotion_entailment",
    image_name: "reccon-emotion-entailment",
    deployment_name: "emotion-entailment"
  },
  "lsr": {
    module_name: "lsr",
    image_name: "lsr",
    deployment_name: "lsr"
  },
  "span_extraction": {
    module_name: "span_extraction",
    image_name: "reccon-span-extraction",
    deployment_name: "span-extraction"
  },
  "rumour_detection_twitter": {
    module_name: "rumour_detection_twitter",
    image_name: "rumour-detection-twitter",
    deployment_name: "rumour-detection-twitter"
  },
  "csgec": {
    module_name: "csgec",
    image_name: "csgec",
    deployment_name: "csgec"
  },
  "rst_pointer": {
    module_name: "rst_pointer",
    image_name: "rst-pointer",
    deployment_name: "rst-pointer"
  },
  "lif_3way_ap": {
    module_name: "lif_3way_ap",
    image_name: "lif-3way-ap",
    deployment_name: "lif-3way-ap"
  },
  "ufd": {
    module_name: "ufd",
    image_name: "ufd",
    deployment_name: "ufd"
  },
  "sentic_gcn": {
    module_name: "sentic_gcn",
    image_name: "sentic-gcn",
    deployment_name: "sentic-gcn"
  }
};

{
  "stages": [
    "build_and_push_staging",
    "restart_kubernetes_staging",
    "retag_and_push_production",
    "restart_kubernetes_production"
  ],
} + {
  // Build and push staging
  [api_names[key]["module_name"] + "_build_and_push_staging"]:
    build_and_push_staging(api_names[key]["module_name"], api_names[key]["image_name"])
    for key in std.objectFields(api_names)
} + {
  // Restart kubernetes staging
  [api_names[key]["module_name"] + "_restart_kubernetes_staging"]:
    restart_kubernetes_staging(api_names[key]["module_name"], api_names[key]["deployment_name"])
    for key in std.objectFields(api_names)
} + {
  // Retag and push production
  [api_names[key]["module_name"] + "_retag_and_push_production"]:
    retag_and_push_production(api_names[key]["module_name"], api_names[key]["image_name"])
    for key in std.objectFields(api_names)
} + {
  // Restart kubernetes production
  [api_names[key]["module_name"] + "_restart_kubernetes_production"]:
    restart_kubernetes_production(api_names[key]["module_name"], api_names[key]["deployment_name"])
    for key in std.objectFields(api_names)
} + {
  // Docs
  "docs_build_and_push_staging": build_and_push_docs_staging(),
  "docs_restart_kubernetes_staging": restart_kubernetes_staging("docs", "docs"),
  "docs_retag_and_push_production": retag_and_push_production("docs", "docs"),
  "docs_restart_kubernetes_production": restart_kubernetes_production("docs", "docs")
}