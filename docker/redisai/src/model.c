#include "model.h"
#include "model_struct.h"
#include "backends.h"

#include "rmutil/alloc.h"
#include "util/arr_rm_alloc.h"

RedisModuleType *RedisAI_ModelType = NULL;

static void* RAI_Model_RdbLoad(struct RedisModuleIO *io, int encver) {
  // if (encver != RAI_ENC_VER) {
  //   /* We should actually log an error here, or try to implement
  //      the ability to load older versions of our data structure. */
  //   return NULL;
  // }

  RAI_Backend backend = RedisModule_LoadUnsigned(io);
  const char *devicestr = RedisModule_LoadStringBuffer(io, NULL);
  size_t ninputs = RedisModule_LoadUnsigned(io);
  const char **inputs = RedisModule_Alloc(ninputs * sizeof(char*));

  for (size_t i=0; i<ninputs; i++) {
    inputs[i] = RedisModule_LoadStringBuffer(io, NULL);
  }

  size_t noutputs = RedisModule_LoadUnsigned(io);

  const char **outputs = RedisModule_Alloc(ninputs * sizeof(char*));

  for (size_t i=0; i<noutputs; i++) {
    outputs[i] = RedisModule_LoadStringBuffer(io, NULL);
  }

  size_t len;

  char *buffer = RedisModule_LoadStringBuffer(io, &len);

  RAI_Error err = {0};

  RAI_Model *model = RAI_ModelCreate(backend, devicestr, ninputs, inputs, noutputs, outputs,
                                     buffer, len, &err);

  if (err.code == RAI_EBACKENDNOTLOADED) {
    RedisModuleCtx* ctx = RedisModule_GetContextFromIO(io);
    int ret = RAI_LoadDefaultBackend(ctx, backend);
    if (ret == REDISMODULE_ERR) {
      RedisModule_Log(ctx, "error", "Could not load default backend\n");
      RAI_ClearError(&err);
      return NULL;
    }
    RAI_ClearError(&err);
    model = RAI_ModelCreate(backend, devicestr, ninputs, inputs, noutputs, outputs, buffer, len, &err);
  }
 
  if (err.code != RAI_OK) {
    printf("ERR: %s\n", err.detail);
    RAI_ClearError(&err);
    if (buffer) {
      RedisModule_Free(buffer);
    }
    return NULL;
  }

  RedisModule_Free(inputs);
  RedisModule_Free(outputs);
  RedisModule_Free(buffer);

  return model;
}

static void RAI_Model_RdbSave(RedisModuleIO *io, void *value) {
  RAI_Model *model = (RAI_Model*)value;
  char *buffer = NULL;
  size_t len = 0;
  RAI_Error err = {0};

  int ret = RAI_ModelSerialize(model, &buffer, &len, &err);

  if (err.code != RAI_OK) {
    printf("ERR: %s\n", err.detail);
    RAI_ClearError(&err);
    if (buffer) {
      RedisModule_Free(buffer);
    }
    return;
  }

  RedisModule_SaveUnsigned(io, model->backend);
  RedisModule_SaveStringBuffer(io, model->devicestr, strlen(model->devicestr) + 1);
  RedisModule_SaveUnsigned(io, model->ninputs);
  for (size_t i=0; i<model->ninputs; i++) {
    RedisModule_SaveStringBuffer(io, model->inputs[i], strlen(model->inputs[i]) + 1);
  }
  RedisModule_SaveUnsigned(io, model->noutputs);
  for (size_t i=0; i<model->noutputs; i++) {
    RedisModule_SaveStringBuffer(io, model->outputs[i], strlen(model->outputs[i]) + 1);
  }
  RedisModule_SaveStringBuffer(io, buffer, len);

  if (buffer) {
    RedisModule_Free(buffer);
  }
}

static void RAI_Model_AofRewrite(RedisModuleIO *aof, RedisModuleString *key, void *value) {
  RAI_Model *model = (RAI_Model*)value;

  char *buffer = NULL;
  size_t len = 0;
  RAI_Error err = {0};

  int ret = RAI_ModelSerialize(model, &buffer, &len, &err);

  if (err.code != RAI_OK) {
    printf("ERR: %s\n", err.detail);
    RAI_ClearError(&err);
    if (buffer) {
      RedisModule_Free(buffer);
    }
    return;
  }

  // AI.MODELSET model_key backend device [INPUTS name1 name2 ... OUTPUTS name1 name2 ...] model_blob

  RedisModuleString **inputs_ = array_new(RedisModuleString*, model->ninputs);
  RedisModuleString **outputs_ = array_new(RedisModuleString*, model->noutputs);

  RedisModuleCtx *ctx = RedisModule_GetContextFromIO(aof);

  for (size_t i=0; i<model->ninputs; i++) {
    array_append(inputs_, RedisModule_CreateString(ctx, model->inputs[i], strlen(model->inputs[i])));
  }

  for (size_t i=0; i<model->noutputs; i++) {
    array_append(outputs_, RedisModule_CreateString(ctx, model->outputs[i], strlen(model->outputs[i])));
  }

  char backend[256] = "";
  switch (model->backend) {
    case RAI_BACKEND_TENSORFLOW:
      strcpy(backend, "TF");
      break;
    case RAI_BACKEND_TORCH:
      strcpy(backend, "TORCH");
      break;
    case RAI_BACKEND_ONNXRUNTIME:
      strcpy(backend, "ONNX");
      break;
  }

  RedisModule_EmitAOF(aof, "AI.MODELSET", "slccvcvb",
                      key,
                      backend, model->devicestr,
                      "INPUTS", inputs_, model->ninputs,
                      "OUTPUTS", outputs_, model->noutputs,
                      buffer, len);

  for (size_t i=0; i<model->ninputs; i++) {
    RedisModule_FreeString(ctx, inputs_[i]);
  }

  array_free(inputs_);

  for (size_t i=0; i<model->noutputs; i++) {
    RedisModule_FreeString(ctx, outputs_[i]);
  }

  array_free(outputs_);
}

// TODO: pass err in?
static void RAI_Model_DTFree(void *value) {
  RAI_Error err = {0};
  RAI_ModelFree(value, &err);
  if (err.code != RAI_OK) {
    printf("ERR: %s\n", err.detail);
    RAI_ClearError(&err);
  }
}

int RAI_ModelInit(RedisModuleCtx* ctx) {
  RedisModuleTypeMethods tmModel = {
      .version = REDISMODULE_TYPE_METHOD_VERSION,
      .rdb_load = RAI_Model_RdbLoad,
      .rdb_save = RAI_Model_RdbSave,
      .aof_rewrite = RAI_Model_AofRewrite,
      .mem_usage = NULL,
      .free = RAI_Model_DTFree,
      .digest = NULL
  };

  RedisAI_ModelType = RedisModule_CreateDataType(ctx, "AI__MODEL", 0, &tmModel);
  return RedisAI_ModelType != NULL;
}

RAI_Model *RAI_ModelCreate(RAI_Backend backend, const char* devicestr,
                           size_t ninputs, const char **inputs,
                           size_t noutputs, const char **outputs,
                           const char *modeldef, size_t modellen, RAI_Error* err) {
  RAI_Model *model;
  if (backend == RAI_BACKEND_TENSORFLOW) {
    if (!RAI_backends.tf.model_create_with_nodes) {
      RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: TF.\n");
      return NULL;
    }
    model = RAI_backends.tf.model_create_with_nodes(backend, devicestr, ninputs, inputs, noutputs, outputs, modeldef, modellen, err);
  }
  else if (backend == RAI_BACKEND_TORCH) {
    if (!RAI_backends.torch.model_create) {
      RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: TORCH.\n");
      return NULL;
    }
    model = RAI_backends.torch.model_create(backend, devicestr, modeldef, modellen, err);
  }
  else if (backend == RAI_BACKEND_ONNXRUNTIME) {
    if (!RAI_backends.onnx.model_create) {
      RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: ONNX.\n");
      return NULL;
    }
    model = RAI_backends.onnx.model_create(backend, devicestr, modeldef, modellen, err);
  }
  else {
    RAI_SetError(err, RAI_EUNSUPPORTEDBACKEND, "Unsupported backend.\n");
    return NULL;
  }
  return model;
}

void RAI_ModelFree(RAI_Model* model, RAI_Error* err) {
  if (--model->refCount > 0){
    return;
  }

  if (model->backend == RAI_BACKEND_TENSORFLOW) {
    if (!RAI_backends.tf.model_free) {
      RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: TF.\n");
      return;
    }
    RAI_backends.tf.model_free(model, err);
  }
  else if (model->backend == RAI_BACKEND_TORCH) {
    if (!RAI_backends.torch.model_free) {
      RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: TORCH.\n");
      return;
    }
    RAI_backends.torch.model_free(model, err);
  }
  else if (model->backend == RAI_BACKEND_ONNXRUNTIME) {
    if (!RAI_backends.onnx.model_free) {
      RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: ONNX.\n");
      return;
    }
    RAI_backends.onnx.model_free(model, err);
  }
  else {
    RAI_SetError(err, RAI_EUNSUPPORTEDBACKEND, "Unsupported backend\n");
    return;
  }

  RedisModule_Free(model);
}

RAI_ModelRunCtx* RAI_ModelRunCtxCreate(RAI_Model* model) {
#define PARAM_INITIAL_SIZE 10
  RAI_ModelRunCtx* mctx = RedisModule_Calloc(1, sizeof(*mctx));
  mctx->model = RAI_ModelGetShallowCopy(model);
  mctx->inputs = array_new(RAI_ModelCtxParam, PARAM_INITIAL_SIZE);
  mctx->outputs = array_new(RAI_ModelCtxParam, PARAM_INITIAL_SIZE);
  return mctx;
}

static int Model_RunCtxAddParam(RAI_ModelRunCtx* mctx, RAI_ModelCtxParam** paramArr,
                                const char* name, RAI_Tensor* tensor) {

  RAI_ModelCtxParam param = {
      .name = name,
      .tensor = tensor ? RAI_TensorGetShallowCopy(tensor): NULL,
  };
  *paramArr = array_append(*paramArr, param);
  return 1;
}

int RAI_ModelRunCtxAddInput(RAI_ModelRunCtx* mctx, const char* inputName, RAI_Tensor* inputTensor) {
  return Model_RunCtxAddParam(mctx, &mctx->inputs, inputName, inputTensor);
}

int RAI_ModelRunCtxAddOutput(RAI_ModelRunCtx* mctx, const char* outputName) {
  return Model_RunCtxAddParam(mctx, &mctx->outputs, outputName, NULL);
}

size_t RAI_ModelRunCtxNumOutputs(RAI_ModelRunCtx* mctx) {
  return array_len(mctx->outputs);
}

RAI_Tensor* RAI_ModelRunCtxOutputTensor(RAI_ModelRunCtx* mctx, size_t index) {
  assert(RAI_ModelRunCtxNumOutputs(mctx) > index && index >= 0);
  return mctx->outputs[index].tensor;
}

void RAI_ModelRunCtxFree(RAI_ModelRunCtx* mctx) {
  for (size_t i = 0 ; i < array_len(mctx->inputs) ; ++i) {
    RAI_TensorFree(mctx->inputs[i].tensor);
  }
  array_free(mctx->inputs);

  for (size_t i = 0 ; i < array_len(mctx->outputs) ; ++i) {
    if (mctx->outputs[i].tensor) {
      RAI_TensorFree(mctx->outputs[i].tensor);
    }
  }
  array_free(mctx->outputs);

  RAI_Error err = {0};
  RAI_ModelFree(mctx->model, &err);

  if (err.code != RAI_OK) {
    // TODO: take it to client somehow
    printf("ERR: %s\n", err.detail);
    RAI_ClearError(&err);
  }

  RedisModule_Free(mctx);
}

int RAI_ModelRun(RAI_ModelRunCtx* mctx, RAI_Error* err) {
  int ret;

  switch (mctx->model->backend) {
    case RAI_BACKEND_TENSORFLOW:
      if (!RAI_backends.tf.model_run) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: TF.\n");
        return REDISMODULE_ERR;
      }
      ret = RAI_backends.tf.model_run(mctx, err);
      break;
    case RAI_BACKEND_TORCH:
      if (!RAI_backends.torch.model_run) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: TORCH.\n");
        return REDISMODULE_ERR;
      }
      ret = RAI_backends.torch.model_run(mctx, err);
      break;
    case RAI_BACKEND_ONNXRUNTIME:
      if (!RAI_backends.onnx.model_run) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: ONNX.\n");
        return REDISMODULE_ERR;
      }
      ret = RAI_backends.onnx.model_run(mctx, err);
      break;
    default:
      RAI_SetError(err, RAI_EUNSUPPORTEDBACKEND, "Unsupported backend.\n");
      return REDISMODULE_ERR;
  }

  return ret;
}

RAI_Model* RAI_ModelGetShallowCopy(RAI_Model* model) {
  ++model->refCount;
  return model;
}

int RAI_ModelSerialize(RAI_Model *model, char **buffer, size_t *len, RAI_Error *err) {
  int ret;

  switch (model->backend) {
    case RAI_BACKEND_TENSORFLOW:
      if (!RAI_backends.tf.model_serialize) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: TF.\n");
        return REDISMODULE_ERR;
      }
      ret = RAI_backends.tf.model_serialize(model, buffer, len, err);
      break;
    case RAI_BACKEND_TORCH:
      if (!RAI_backends.torch.model_serialize) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: TORCH.\n");
        return REDISMODULE_ERR;
      }
      ret = RAI_backends.torch.model_serialize(model, buffer, len, err);
      break;
    case RAI_BACKEND_ONNXRUNTIME:
      if (!RAI_backends.onnx.model_serialize) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "Backend not loaded: ONNX.\n");
        return REDISMODULE_ERR;
      }
      ret = RAI_backends.onnx.model_serialize(model, buffer, len, err);
      break;
    default:
      RAI_SetError(err, RAI_EUNSUPPORTEDBACKEND, "Unsupported backend.\n");
      return REDISMODULE_ERR;
  }

  return ret;
}
