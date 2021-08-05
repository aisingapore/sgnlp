local build_and_push_staging(module_name, image_name) = {
  image: "registry.aisingapore.net/sg-nlp/sg-nlp-runner:latest",
  stage: "build_and_push_staging",
  when: "manual",
  script: [
    "echo 'Logging in to AISG Docker Registry...'",
    "echo $STG_REGISTRY_PASSWORD | docker login registry.aisingapore.net -u $STG_DOCKER_USER --password-stdin",
    "cd demo_api/%s" % module_name,
    "echo 'Building and pushing image...'",
    "docker build --no-cache -t %s ." % module_name,
    "docker tag %s registry.aisingapore.net/sg-nlp/%s:latest" % [module_name, image_name],
    "docker push registry.aisingapore.net/sg-nlp/%s:latest" % image_name,
  ],
};

local retag_and_push_production(module_name, image_name) = {
  image: "registry.aisingapore.net/sg-nlp/sg-nlp-runner:latest",
  stage: "retag_and_push_production",
  only: {
    refs: ["main"]
  },
  needs: ["%s_restart_kubernetes_staging" % module_name],
  when: "manual",
  script: [
    "echo 'Logging in to AISG Docker Registry...'",
    "echo $STG_REGISTRY_PASSWORD | docker login registry.aisingapore.net -u $STG_DOCKER_USER --password-stdin",
    "docker pull registry.aisingapore.net/sg-nlp/%s:latest" % [image_name],
    "echo 'Logging in to AKS Registry'",
    "echo $PROD_REGISTRY_PASSWORD | docker login aisgk8sregistry.azurecr.io -u $PROD_REGISTRY_USER --password-stdin",
    "echo 'Retagging and pushing image...'",
    "docker tag registry.aisingapore.net/sg-nlp/%s:latest aisgk8sregistry.azurecr.io/sgnlp/%s:latest" % [image_name, image_name],
    "docker push aisgk8sregistry.azurecr.io/sgnlp/%s:latest" % image_name,
  ],
};

local restart_kubernetes_staging(module_name, deployment_name) = {
  image: "registry.aisingapore.net/sea-core-nlp/seacorenlp-runner:latest",
  stage: "restart_kubernetes_staging",
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
  only: {
    refs: ["main"]
  },
  when: "manual",
  needs: ["%s_retag_and_push_production" % module_name],
  script: [
    "echo 'Restarting pods...'",
    "export KUBECONFIG=$PROD_KUBE_CONFIG",
    "kubectl rollout restart deployment/%s -n sgnlp" % deployment_name
  ]
};

local api_names = {
  "emotion_entailment": {
    module_name: "emotion_entailment",
    image_name: "reccon-emotion-entailment",
    deployment_name: "emotion-entailment"
  },
  "lif_3way_ap": {
    module_name: "lif_3way_ap",
    image_name: "lif-3way-ap",
    deployment_name: "lif-3way-ap"
  },
  "lsr": {
    module_name: "lsr",
    image_name: "lsr",
    deployment_name: "lsr"
  },
  "nea": {
    module_name: "nea",
    image_name: "nea",
    deployment_name: "nea"
  },
  "span_extraction": {
    module_name: "span_extraction",
    image_name: "reccon-span-extraction",
    deployment_name: "span-extraction"
  },
  "ufd": {
    module_name: "ufd",
    image_name: "ufd",
    deployment_name: "ufd"
  },
  "rumour_detection_twitter": {
    module_name: "rumour_detection_twitter",
    image_name: "rumour-detection-twitter",
    deployment_name: "rumour-detection-twitter"
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
}