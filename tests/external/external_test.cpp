#include <OpenImageDenoise/oidn.h>
#include <cstdio>
int main()
{
  std::printf("OIDN %d with the Light Tracer Dawn provider\n", OIDN_VERSION);
  return !oidnIsWebGPUDeviceSupported();
}
