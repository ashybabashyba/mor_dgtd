#include "driver.h"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Error: Falta archivo de configuración.\n";
        std::cerr << "Uso: " << argv[0] << " <ruta_al_archivo_config.json>\n";
        return 1;
    }

    std::string json_path = argv[1];

    try {
        std::cout << "==========================================\n";
        std::cout << "      LAUNCHER: DGTD MODEL ORDER REDUCTION\n";
        std::cout << "==========================================\n";
        
        // Ceder el control por completo al driver
        mor::driver::run_simulation(json_path);
        
        std::cout << "==========================================\n";
        std::cout << "      PROCESO FINALIZADO EXITOSAMENTE     \n";
        std::cout << "==========================================\n";
    } 
    catch (const std::exception& e) {
        std::cerr << "\n[ERROR CRÍTICO EJECUCIÓN]: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}