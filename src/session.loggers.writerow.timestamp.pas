unit session.loggers.writerow.timestamp;

{$mode ObjFPC}{$H+}

interface

uses
  session.loggers.instances;

procedure Timestamp(AEvent : string);

var
  SaveData : TDataProcedure;

implementation

uses SysUtils, session.pool, timestamps, session.loggers;

var
  BaseHeader : string;

const
  Tab = #9;

procedure InitializeBaseHeader;
begin
  BaseHeader := TLogger.Row([
    'Timestamp',
    'Session.Trial.UID',
    'Session.Block.UID',
    'Event',
    'Event.Annotation']);
end;

procedure Timestamp(AEvent : string);
var
  LTimestamp : string;
begin
  LTimestamp := TimestampToStr(TickCount - Pool.TimeStart);
  SaveData(TLogger.Row([BaseHeader+
    LTimestamp,
    (Pool.Session.Trial.UID + 1).ToString,
    (Pool.Session.Block.UID + 1).ToString,
    AEvent]));
end;

end.

