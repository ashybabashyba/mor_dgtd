#pragma once
#include <string>

namespace mor::driver {
    // Función principal que ejecutará el flujo completo basado en el JSON
    void run_simulation(const std::string& json_path);
}