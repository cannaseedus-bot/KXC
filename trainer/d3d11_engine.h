#pragma once
// d3d11_engine.h — Trainer-local D3D11 engine.
// Exposes rawDevice() / rawCtx() for direct buffer management in GPT2Trainer.
// NOT the XVM d3d11_engine (which has uploadVM/dispatch/xvm_core.h dependency).
#include <d3d11.h>
#include <wrl/client.h>
#include <string>

using Microsoft::WRL::ComPtr;

class D3D11Engine {
public:
    bool init(bool forceWarp = false, bool verboseLog = false);

    bool               usedWarp()     const { return usedWarp_; }
    D3D_FEATURE_LEVEL  featureLevel() const { return featureLevel_; }
    const std::string& adapterName()  const { return adapterName_; }
    const std::string& initReason()   const { return initReason_; }

    ID3D11Device*        rawDevice() { return device_.Get(); }
    ID3D11DeviceContext* rawCtx()    { return ctx_.Get(); }

private:
    ComPtr<ID3D11Device>        device_;
    ComPtr<ID3D11DeviceContext> ctx_;
    bool              usedWarp_     = false;
    D3D_FEATURE_LEVEL featureLevel_ = D3D_FEATURE_LEVEL_11_0;
    std::string       adapterName_;
    std::string       initReason_;
};
