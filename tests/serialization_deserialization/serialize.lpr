program serialize;

uses
  Interfaces,
  Classes, SysUtils,
  session.counters.all;

var
  Counter : TSessionCounters;
  Filename : string;
  Before: string;
  After : string;

begin
  Filename := 'string.bin';
  Counter := TSessionCounters.Create(True);
  Counter.Next;
  Counter.Next;
  Counter.NextBlockID(10);
  Counter.NextTrialID(4);
  Counter.Trial.Events.Hit;
  Counter.Block.Events.Miss;
  Counter.Block.Trial.Events.None;
  Counter.Block.Trial.Events.Hit;
  Counter.Events.None;

  WriteLn('Writing');
  Before := Counter.ToIni;
  WriteLn(Before);
  try
    Counter.SaveToFile(Filename);
  finally
    Counter.Free;
  end;
  Write(StringOfChar(LineEnding[1], Length(LineEnding)*3));

  Counter := TSessionCounters.Create(True);
  try
    Counter.LoadFromFile(FileName);
    WriteLn('Loading');
    After := Counter.ToIni;
    WriteLn(After);
  finally
    Counter.Free;
  end;

  Write(StringOfChar(LineEnding[1], Length(LineEnding)*3));
  WriteLn(Before = After);
  ReadLn;
end.

