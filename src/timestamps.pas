{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit timestamps;

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Math

{$IFDEF LINUX}
  , Linux
  , UnixType
{$ENDIF}

{$IFDEF DARWIN}
  , ctypes
  , timestamps.machtime
{$ENDIF}

{$IFDEF WINDOWS}
  , Windows
{$ENDIF}
  ;

function ClockMonotonic : Float;

implementation

{$IFDEF LINUX}
function ClockMonotonic: Float;
var
  tp: timespec;
  a, b : Float;
begin
  clock_gettime(CLOCK_MONOTONIC, @tp);
  a := Float(tp.tv_sec);
  b := Float(tp.tv_nsec) * 1e-9;
  Result := a+b;
end;
{$ENDIF}

{$IFDEF DARWIN}
{credits: https://github.com/pupil-labs/pyuvc/blob/master/pyuvc-source/darwin_time.pxi}

var
  timeConvert: Float = 0.0;

//function get_sys_time_monotonic: Float;
function ClockMonotonic : Float;
var
  timeBase: mach_timebase_info_data_t;
begin
  if timeConvert = 0.0 then begin
    mach_timebase_info(@timeBase);
    timeConvert :=
      (Float(timeBase.numer) / Float(timeBase.denom) / Float(1000000000.0);
  end;
  Result := mach_absolute_time() * timeConvert;
end;
{$ENDIF}

{$IFDEF WINDOWS}
var
  PerSecond : TLargeInteger;

function ClockMonotonic: Float;
var
  Count : TLargeInteger;
begin
  QueryPerformanceCounter(Count);
  Result := Float(Count) / Float(PerSecond);
end;


initialization
   QueryPerformanceFrequency(PerSecond);
{$ENDIF}

end.

