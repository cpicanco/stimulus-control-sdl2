{

Test suit for timing experiments using threads.

Aimed at monotonic timing.

Initializes timing references, sets the main thread's CPU affinity, and then
runs two parallel sets of loops(one in the main thread, one in TMyThread1).
Both loops repeatedly measure and output elapsed timesbased on clock_monotonic
relative to an initial start time,and also include 1-second sleep intervals.
The worker thread uses the main thread's initial clock_monotonic timeas its
reference. Both threads also log their total operational durationusing both
TDateTime and clock_monotonic.

}

program windows_timing;

uses Classes, SysUtils, debug.output, timestamps.windows, Windows;

type
  TimeBufferRange = 0..99;

var
  Range : TimeBufferRange;

function DateTimeToStrUs(dt: TDatetime): string;
var
  us: string;
begin
  //Spit out most of the result: '20160802 11:34:36.'
  Result := FormatDateTime('yyyymmdd hh":"nn":"ss"."', dt);

  //extract the number of microseconds
  dt := Frac(dt); //fractional part of day
  dt := dt * 24*60*60; //number of seconds in that day
  us := IntToStr(Round(Frac(dt)*1000000));

  //Add the us integer to the end:
  // '20160801 11:34:36.' + '00' + '123456'
  Result := Result + StringOfChar('0', 6-Length(us)) + us;
end;

function Elapsed(AStopTime, AStartTime: Double) : string;
begin
  Result := FloatToStrF(AStopTime - AStartTime, ffFixed, 0, 9);
end;

type

  { TMyThread1 }

  TMyThread1 = class(TThread)
  protected
    procedure Execute; override;
  public
    FStartTime : Double;
    constructor Create;
  end;

  constructor TMyThread1.Create;
  begin
    inherited Create(True);

  end;

  procedure TMyThread1.Execute;
  var
    i, j : integer;
    LStartTime: Double;
    LStartRealTime: TDateTime;
    LStopTime: Double;
    LStopRealTime: TDateTime;
    //LTimeBuffer : array[0..99] of Double;
  begin
    WriteLn('Thread 1 is running');
    Print1('Thread 1 is running');

    LStartRealTime := Now;
    LStartTime := FStartTime;

    WriteLn(DateTimeToStrUs(LStartRealTime));
    Print1(DateTimeToStrUs(LStartRealTime));

    for j := 0 to 10 do begin
      for i := Low(Range) to High(Range) do begin
        WriteLn(Elapsed(clock_monotonic, LStartTime)+'T');
        //LTimeBuffer[i] := clock_monotonic;
      end;

      //for i := Low(LTimeBuffer) to High(LTimeBuffer) do begin
      //  WriteLn(Elapsed(LTimeBuffer[i], LStartTime)+'T*');
      //  Print1(Elapsed(LTimeBuffer[i], LStartTime));
      //end;

      Sleep(1000);
    end;
    LStopRealTime := Now;
    LStopTime := clock_monotonic;

    WriteLn(DateTimeToStrUs((LStopRealTime - LStartRealTime)));
    Print1(DateTimeToStrUs((LStopRealTime - LStartRealTime)));

    WriteLn(Elapsed(LStopTime, LStartTime));
    Print1(Elapsed(LStopTime, LStartTime));
  end;

var
  MyThread1: TMyThread1;
  StartTime: Double;
  StartRealTime: TDateTime;
  StopTime: Double;
  StopRealTime: TDateTime;

  i, j : Integer;
  //TimeBuffer : array[TimeBufferRange] of Double;
begin
  StartRealTime := Now;
  StartTime := clock_monotonic;

  SetThreadAffinityMask(
    OpenThread(THREAD_ALL_ACCESS, False, GetCurrentThreadId), 1);

  WriteLn('Main thread is running');
  Print2('Main thread is running');

  MyThread1 := TMyThread1.Create;
  MyThread1.FStartTime := StartTime;
  MyThread1.Start;

  WriteLn(DateTimeToStrUs(StartRealTime));
  Print2(DateTimeToStrUs(StartRealTime));

  for j := 0 to 10 do begin
    for i := Low(Range) to High(Range) do begin
      WriteLn(Elapsed(clock_monotonic, StartTime));
      //TimeBuffer[i] := clock_monotonic;
    end;

    //for i := Low(TimeBuffer) to High(TimeBuffer) do begin
    //  WriteLn(Elapsed(TimeBuffer[i], StartTime)+'*');
    //  Print2(Elapsed(TimeBuffer[i], StartTime));
    //end;

    Sleep(1000);
  end;
  StopRealTime := Now;
  StopTime := clock_monotonic;

  WriteLn(DateTimeToStrUs((StopRealTime - StartRealTime)));
  Print2(DateTimeToStrUs((StopRealTime - StartRealTime)));

  WriteLn(Elapsed(StopTime, StartTime));
  Print2(Elapsed(StopTime, StartTime));

  // Wait for the thread1 to finish
  MyThread1.WaitFor;
  ReadLn;
end.

