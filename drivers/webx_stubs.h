#pragma once
//
// webx_stubs.h — Minimal WebX type stubs for WASM compilation
//
// Replaces the full webx_compute.h (which pulls in Windows headers, GPU backends,
// field graphs, BOSS scheduling, etc.) with the bare minimum types TaskEngine
// actually references: Provider, ProviderManager.
//

#include <string>
#include <vector>

namespace WebX {

struct Provider {
    std::string id;
    std::string library;
    bool available = true;
};

class ProviderManager {
public:
    explicit ProviderManager(std::vector<Provider> providers)
        : providers_(std::move(providers)) {}

    const std::vector<Provider>& getProviders() const { return providers_; }

private:
    std::vector<Provider> providers_;
};

} // namespace WebX
